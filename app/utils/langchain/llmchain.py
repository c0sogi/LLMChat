"""Chain that just formats a prompt and calls an LLM."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Callable

import langchain
import yaml
from langchain.callbacks import get_callback_manager
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.conversation.prompt import PROMPT
from langchain.input import get_colored_text
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BaseLanguageModel, LLMResult, PromptValue
from pydantic import BaseModel, Extra, Field, root_validator, validator

from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from app.utils.langchain.vectorstore_memory import BaseMemory, VectorStoreRetrieverMemory


from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import BaseMessage, BaseRetriever, Document

# Depending on the memory type and configuration, the chat history format may differ.
# This needs to be consolidated.
CHAT_TURN_TYPE = Union[Tuple[str, str], BaseMessage]


_ROLE_MAP = {"human": "Human: ", "ai": "Assistant: "}


def _get_chat_history(chat_history: List[CHAT_TURN_TYPE]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        if isinstance(dialogue_turn, BaseMessage):
            role_prefix = _ROLE_MAP.get(dialogue_turn.type, f"{dialogue_turn.type}: ")
            buffer += f"\n{role_prefix}{dialogue_turn.content}"
        elif isinstance(dialogue_turn, tuple):
            human = "Human: " + dialogue_turn[0]
            ai = "Assistant: " + dialogue_turn[1]
            buffer += "\n" + "\n".join([human, ai])
        else:
            raise ValueError(
                f"Unsupported chat history format: {type(dialogue_turn)}." f" Full chat history: {chat_history} "
            )
    return buffer


def _get_verbosity() -> bool:
    return langchain.verbose


class Chain(BaseModel, ABC):
    """Base interface that all chains should implement."""

    memory: Optional[BaseMemory] = None
    callback_manager: BaseCallbackManager = Field(default_factory=get_callback_manager, exclude=True)
    verbose: bool = Field(default_factory=_get_verbosity)  # Whether to print the response text

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    def _chain_type(self) -> str:
        raise NotImplementedError("Saving not supported for this chain type.")

    @validator("callback_manager", pre=True, always=True)
    def set_callback_manager(cls, callback_manager: Optional[BaseCallbackManager]) -> BaseCallbackManager:
        """If callback manager is None, set it.

        This allows users to pass in None as callback manager, which is a nice UX.
        """
        return callback_manager or get_callback_manager()

    @validator("verbose", pre=True, always=True)
    def set_verbose(cls, verbose: Optional[bool]) -> bool:
        """If verbose is None, set it.

        This allows users to pass in None as verbose to access the global setting.
        """
        if verbose is None:
            return _get_verbosity()
        else:
            return verbose

    @property
    @abstractmethod
    def input_keys(self) -> List[str]:
        """Input keys this chain expects."""

    @property
    @abstractmethod
    def output_keys(self) -> List[str]:
        """Output keys this chain expects."""

    def _validate_inputs(self, inputs: Dict[str, str]) -> None:
        """Check that all inputs are present."""
        missing_keys = set(self.input_keys).difference(inputs)
        if missing_keys:
            raise ValueError(f"Missing some input keys: {missing_keys}")

    def _validate_outputs(self, outputs: Dict[str, str]) -> None:
        if set(outputs) != set(self.output_keys):
            raise ValueError(
                f"Did not get output keys that were expected. "
                f"Got: {set(outputs)}. Expected: {set(self.output_keys)}."
            )

    @abstractmethod
    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Run the logic of this chain and return the output."""

    async def _acall(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Run the logic of this chain and return the output."""
        raise NotImplementedError("Async call not supported for this chain type.")

    def __call__(self, inputs: Union[Dict[str, Any], Any], return_only_outputs: bool = False) -> Dict[str, Any]:
        """Run the logic of this chain and add to output if desired.

        Args:
            inputs: Dictionary of inputs, or single input if chain expects
                only one param.
            return_only_outputs: boolean for whether to return only outputs in the
                response. If True, only new keys generated by this chain will be
                returned. If False, both input keys and new keys generated by this
                chain will be returned. Defaults to False.

        """
        inputs = self.prep_inputs(inputs)
        self.callback_manager.on_chain_start(
            {"name": self.__class__.__name__},
            inputs,
            verbose=self.verbose,
        )
        try:
            outputs = self._call(inputs)
        except (KeyboardInterrupt, Exception) as e:
            self.callback_manager.on_chain_error(e, verbose=self.verbose)
            raise e
        self.callback_manager.on_chain_end(outputs, verbose=self.verbose)
        return self.prep_outputs(inputs, outputs, return_only_outputs)

    async def acall(self, inputs: Union[Dict[str, Any], Any], return_only_outputs: bool = False) -> Dict[str, Any]:
        """Run the logic of this chain and add to output if desired.

        Args:
            inputs: Dictionary of inputs, or single input if chain expects
                only one param.
            return_only_outputs: boolean for whether to return only outputs in the
                response. If True, only new keys generated by this chain will be
                returned. If False, both input keys and new keys generated by this
                chain will be returned. Defaults to False.

        """
        inputs = await self.aprep_inputs(inputs)
        if self.callback_manager.is_async:
            await self.callback_manager.on_chain_start(
                {"name": self.__class__.__name__},
                inputs,
                verbose=self.verbose,
            )
        else:
            self.callback_manager.on_chain_start(
                {"name": self.__class__.__name__},
                inputs,
                verbose=self.verbose,
            )
        try:
            outputs = await self._acall(inputs)
        except (KeyboardInterrupt, Exception) as e:
            if self.callback_manager.is_async:
                await self.callback_manager.on_chain_error(e, verbose=self.verbose)
            else:
                self.callback_manager.on_chain_error(e, verbose=self.verbose)
            raise e
        if self.callback_manager.is_async:
            await self.callback_manager.on_chain_end(outputs, verbose=self.verbose)
        else:
            self.callback_manager.on_chain_end(outputs, verbose=self.verbose)
        return await self.aprep_outputs(inputs, outputs, return_only_outputs)

    def prep_outputs(
        self,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        return_only_outputs: bool = False,
    ) -> Dict[str, str]:
        """Validate and prep outputs."""
        self._validate_outputs(outputs)
        if self.memory is not None:
            self.memory.save_context(inputs, outputs)
        if return_only_outputs:
            return outputs
        else:
            return {**inputs, **outputs}

    async def aprep_outputs(
        self,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        return_only_outputs: bool = False,
    ) -> Dict[str, str]:
        """Validate and prep outputs."""
        self._validate_outputs(outputs)
        if self.memory is not None:
            await self.memory.asave_context(inputs, outputs)
        if return_only_outputs:
            return outputs
        else:
            return {**inputs, **outputs}

    def prep_inputs(self, inputs: Union[Dict[str, Any], Any]) -> Dict[str, str]:
        """Validate and prep inputs."""
        if not isinstance(inputs, dict):
            _input_keys = set(self.input_keys)
            if self.memory is not None:
                # If there are multiple input keys, but some get set by memory so that
                # only one is not set, we can still figure out which key it is.
                _input_keys = _input_keys.difference(self.memory.memory_variables)
            if len(_input_keys) != 1:
                raise ValueError(
                    f"A single string input was passed in, but this chain expects "
                    f"multiple inputs ({_input_keys}). When a chain expects "
                    f"multiple inputs, please call it by passing in a dictionary, "
                    "eg `chain({'foo': 1, 'bar': 2})`"
                )
            inputs = {list(_input_keys)[0]: inputs}
        if self.memory is not None:
            external_context = self.memory.load_memory_variables(inputs)
            inputs = dict(inputs, **external_context)
        self._validate_inputs(inputs)
        return inputs

    async def aprep_inputs(self, inputs: Union[Dict[str, Any], Any]) -> Dict[str, str]:
        """Validate and prep inputs."""
        if not isinstance(inputs, dict):
            _input_keys = set(self.input_keys)
            if self.memory is not None:
                # If there are multiple input keys, but some get set by memory so that
                # only one is not set, we can still figure out which key it is.
                _input_keys = _input_keys.difference(self.memory.memory_variables)
            if len(_input_keys) != 1:
                raise ValueError(
                    f"A single string input was passed in, but this chain expects "
                    f"multiple inputs ({_input_keys}). When a chain expects "
                    f"multiple inputs, please call it by passing in a dictionary, "
                    "eg `chain({'foo': 1, 'bar': 2})`"
                )
            inputs = {list(_input_keys)[0]: inputs}

        if self.memory is not None:
            external_context = await self.memory.aload_memory_variables(inputs)
            inputs = dict(inputs, **external_context)
        self._validate_inputs(inputs)
        return inputs

    def apply(self, input_list: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Call the chain on all inputs in the list."""
        return [self(inputs) for inputs in input_list]

    def run(self, *args: Any, **kwargs: Any) -> str:
        """Run the chain as text in, text out or multiple variables, text out."""
        if len(self.output_keys) != 1:
            raise ValueError(
                f"`run` not supported when there is not exactly " f"one output key. Got {self.output_keys}."
            )

        if args and not kwargs:
            if len(args) != 1:
                raise ValueError("`run` supports only one positional argument.")
            return self(args[0])[self.output_keys[0]]

        if kwargs and not args:
            return self(kwargs)[self.output_keys[0]]

        raise ValueError(
            f"`run` supported with either positional arguments or keyword arguments"
            f" but not both. Got args: {args} and kwargs: {kwargs}."
        )

    async def arun(self, *args: Any, **kwargs: Any) -> str:
        """Run the chain as text in, text out or multiple variables, text out."""
        if len(self.output_keys) != 1:
            raise ValueError(
                f"`run` not supported when there is not exactly " f"one output key. Got {self.output_keys}."
            )

        if args and not kwargs:
            if len(args) != 1:
                raise ValueError("`run` supports only one positional argument.")
            return (await self.acall(args[0]))[self.output_keys[0]]

        if kwargs and not args:
            return (await self.acall(kwargs))[self.output_keys[0]]

        raise ValueError(
            f"`run` supported with either positional arguments or keyword arguments"
            f" but not both. Got args: {args} and kwargs: {kwargs}."
        )

    def dict(self, **kwargs: Any) -> Dict:
        """Return dictionary representation of chain."""
        if self.memory is not None:
            raise ValueError("Saving of memory is not yet supported.")
        _dict = super().dict()
        _dict["_type"] = self._chain_type
        return _dict

    def save(self, file_path: Union[Path, str]) -> None:
        """Save the chain.

        Args:
            file_path: Path to file to save the chain to.

        Example:
        .. code-block:: python

            chain.save(file_path="path/chain.yaml")
        """
        # Convert file to Path object.
        if isinstance(file_path, str):
            save_path = Path(file_path)
        else:
            save_path = file_path

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        # Fetch dictionary to save
        chain_dict = self.dict()

        if save_path.suffix == ".json":
            with open(file_path, "w") as f:
                json.dump(chain_dict, f, indent=4)
        elif save_path.suffix == ".yaml":
            with open(file_path, "w") as f:
                yaml.dump(chain_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"{save_path} must be json or yaml")


class LLMChain(Chain):
    """Chain to run queries against LLMs.

    Example:
        .. code-block:: python

            from langchain import LLMChain, OpenAI, PromptTemplate
            prompt_template = "Tell me a {adjective} joke"
            prompt = PromptTemplate(
                input_variables=["adjective"], template=prompt_template
            )
            llm = LLMChain(llm=OpenAI(), prompt=prompt)
    """

    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: BaseLanguageModel
    output_key: str = "text"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        return self.apply([inputs])[0]

    def generate(self, input_list: List[Dict[str, Any]]) -> LLMResult:
        """Generate LLM result from inputs."""
        prompts, stop = self.prep_prompts(input_list)
        return self.llm.generate_prompt(prompts, stop)

    async def agenerate(self, input_list: List[Dict[str, Any]]) -> LLMResult:
        """Generate LLM result from inputs."""
        prompts, stop = await self.aprep_prompts(input_list)
        return await self.llm.agenerate_prompt(prompts, stop)

    def prep_prompts(self, input_list: List[Dict[str, Any]]) -> Tuple[List[PromptValue], Optional[List[str]]]:
        """Prepare prompts from inputs."""
        stop = None
        if "stop" in input_list[0]:
            stop = input_list[0]["stop"]
        prompts = []
        for inputs in input_list:
            selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
            prompt = self.prompt.format_prompt(**selected_inputs)
            _colored_text = get_colored_text(prompt.to_string(), "green")
            _text = "Prompt after formatting:\n" + _colored_text
            self.callback_manager.on_text(_text, end="\n", verbose=self.verbose)
            if "stop" in inputs and inputs["stop"] != stop:
                raise ValueError("If `stop` is present in any inputs, should be present in all.")
            prompts.append(prompt)
        return prompts, stop

    async def aprep_prompts(self, input_list: List[Dict[str, Any]]) -> Tuple[List[PromptValue], Optional[List[str]]]:
        """Prepare prompts from inputs."""
        stop = None
        if "stop" in input_list[0]:
            stop = input_list[0]["stop"]
        prompts = []
        for inputs in input_list:
            selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
            prompt = self.prompt.format_prompt(**selected_inputs)
            _colored_text = get_colored_text(prompt.to_string(), "green")
            _text = "Prompt after formatting:\n" + _colored_text
            if self.callback_manager.is_async:
                await self.callback_manager.on_text(_text, end="\n", verbose=self.verbose)
            else:
                self.callback_manager.on_text(_text, end="\n", verbose=self.verbose)
            if "stop" in inputs and inputs["stop"] != stop:
                raise ValueError("If `stop` is present in any inputs, should be present in all.")
            prompts.append(prompt)
        return prompts, stop

    def apply(self, input_list: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Utilize the LLM generate method for speed gains."""
        response = self.generate(input_list)
        return self.create_outputs(response)

    async def aapply(self, input_list: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Utilize the LLM generate method for speed gains."""
        response = await self.agenerate(input_list)
        return self.create_outputs(response)

    def create_outputs(self, response: LLMResult) -> List[Dict[str, str]]:
        """Create outputs from response."""
        return [
            # Get the text of the top generated string.
            {self.output_key: generation[0].text}
            for generation in response.generations
        ]

    async def _acall(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        return (await self.aapply([inputs]))[0]

    def predict(self, **kwargs: Any) -> str:
        """Format prompt with kwargs and pass to LLM.

        Args:
            **kwargs: Keys to pass to prompt template.

        Returns:
            Completion from LLM.

        Example:
            .. code-block:: python

                completion = llm.predict(adjective="funny")
        """
        return self(kwargs)[self.output_key]

    async def apredict(self, **kwargs: Any) -> str:
        """Format prompt with kwargs and pass to LLM.

        Args:
            **kwargs: Keys to pass to prompt template.

        Returns:
            Completion from LLM.

        Example:
            .. code-block:: python

                completion = llm.predict(adjective="funny")
        """
        return (await self.acall(kwargs))[self.output_key]

    def predict_and_parse(self, **kwargs: Any) -> Union[str, List[str], Dict[str, str]]:
        """Call predict and then parse the results."""
        result = self.predict(**kwargs)
        if self.prompt.output_parser is not None:
            return self.prompt.output_parser.parse(result)
        else:
            return result

    async def apredict_and_parse(self, **kwargs: Any) -> Union[str, List[str], Dict[str, str]]:
        """Call apredict and then parse the results."""
        result = await self.apredict(**kwargs)
        if self.prompt.output_parser is not None:
            return self.prompt.output_parser.parse(result)
        else:
            return result

    def apply_and_parse(self, input_list: List[Dict[str, Any]]) -> Sequence[Union[str, List[str], Dict[str, str]]]:
        """Call apply and then parse the results."""
        result = self.apply(input_list)
        return self._parse_result(result)

    def _parse_result(self, result: List[Dict[str, str]]) -> Sequence[Union[str, List[str], Dict[str, str]]]:
        if self.prompt.output_parser is not None:
            return [self.prompt.output_parser.parse(res[self.output_key]) for res in result]
        else:
            return result

    async def aapply_and_parse(
        self, input_list: List[Dict[str, Any]]
    ) -> Sequence[Union[str, List[str], Dict[str, str]]]:
        """Call apply and then parse the results."""
        result = await self.aapply(input_list)
        return self._parse_result(result)

    @property
    def _chain_type(self) -> str:
        return "llm_chain"

    @classmethod
    def from_string(cls, llm: BaseLanguageModel, template: str) -> Chain:
        """Create LLMChain from LLM and template."""
        prompt_template = PromptTemplate.from_template(template)
        return cls(llm=llm, prompt=prompt_template)


class BaseConversationalRetrievalChain(Chain):
    """Chain for chatting with an index."""

    combine_docs_chain: BaseCombineDocumentsChain
    question_generator: LLMChain
    output_key: str = "answer"
    return_source_documents: bool = False
    get_chat_history: Optional[Callable[[CHAT_TURN_TYPE], str]] = None
    """Return the source documents."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    @property
    def input_keys(self) -> List[str]:
        """Input keys."""
        return ["question", "chat_history"]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.

        :meta private:
        """
        _output_keys = [self.output_key]
        if self.return_source_documents:
            _output_keys = _output_keys + ["source_documents"]
        return _output_keys

    @abstractmethod
    def _get_docs(self, question: str, inputs: Dict[str, Any]) -> List[Document]:
        """Get docs."""

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])

        if chat_history_str:
            new_question = self.question_generator.run(question=question, chat_history=chat_history_str)
        else:
            new_question = question
        docs = self._get_docs(new_question, inputs)
        new_inputs = inputs.copy()
        new_inputs["question"] = new_question
        new_inputs["chat_history"] = chat_history_str
        answer = self.combine_docs_chain.run(input_documents=docs, **new_inputs)
        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}

    @abstractmethod
    async def _aget_docs(self, question: str, inputs: Dict[str, Any]) -> List[Document]:
        """Get docs."""

    async def _acall(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])
        if chat_history_str:
            new_question = await self.question_generator.arun(question=question, chat_history=chat_history_str)
        else:
            new_question = question
        docs = await self._aget_docs(new_question, inputs)
        new_inputs = inputs.copy()
        new_inputs["question"] = new_question
        new_inputs["chat_history"] = chat_history_str
        answer = await self.combine_docs_chain.arun(input_documents=docs, **new_inputs)
        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}

    def save(self, file_path: Union[Path, str]) -> None:
        if self.get_chat_history:
            raise ValueError("Chain not savable when `get_chat_history` is not None.")
        super().save(file_path)


class ConversationalRetrievalChain(BaseConversationalRetrievalChain):
    """Chain for chatting with an index."""

    retriever: BaseRetriever
    """Index to connect to."""
    max_tokens_limit: Optional[int] = None
    """If set, restricts the docs to return from store based on tokens, enforced only
    for StuffDocumentChain"""

    def _reduce_tokens_below_limit(self, docs: List[Document]) -> List[Document]:
        num_docs = len(docs)

        if self.max_tokens_limit and isinstance(self.combine_docs_chain, StuffDocumentsChain):
            tokens = [self.combine_docs_chain.llm_chain.llm.get_num_tokens(doc.page_content) for doc in docs]
            token_count = sum(tokens[:num_docs])
            while token_count > self.max_tokens_limit:
                num_docs -= 1
                token_count -= tokens[num_docs]

        return docs[:num_docs]

    def _get_docs(self, question: str, inputs: Dict[str, Any]) -> List[Document]:
        docs = self.retriever.get_relevant_documents(question)
        return self._reduce_tokens_below_limit(docs)

    async def _aget_docs(self, question: str, inputs: Dict[str, Any]) -> List[Document]:
        docs = await self.retriever.aget_relevant_documents(question)
        return self._reduce_tokens_below_limit(docs)

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        retriever: BaseRetriever,
        condense_question_prompt: BasePromptTemplate = CONDENSE_QUESTION_PROMPT,
        chain_type: str = "stuff",
        verbose: bool = False,
        combine_docs_chain_kwargs: Optional[Dict] = None,
        **kwargs: Any,
    ) -> BaseConversationalRetrievalChain:
        """Load chain from LLM."""
        combine_docs_chain_kwargs = combine_docs_chain_kwargs or {}
        doc_chain = load_qa_chain(
            llm,
            chain_type=chain_type,
            verbose=verbose,
            **combine_docs_chain_kwargs,
        )
        condense_question_chain = LLMChain(llm=llm, prompt=condense_question_prompt, verbose=verbose)
        return cls(
            retriever=retriever,
            combine_docs_chain=doc_chain,
            question_generator=condense_question_chain,
            **kwargs,
        )
