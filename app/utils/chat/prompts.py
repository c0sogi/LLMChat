# flake8: noqa

CONTEXT_QUESTION_TMPL_QUERY1 = (
    "Context information is below. \n"
    "---------------------\n"
    "{context}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {question}\n"
)

CONTEXT_QUESTION_TMPL_QUERY2 = (
    "Context information is below. \n"
    "---------------------\n"
    "{context}"
    "\n---------------------\n"
    "answer the question: {question}\n"
)

USER_AI_TMPL_CHAT1 = (
    "The following is a friendly conversation between a {user} and an {ai}. "
    "The {ai} is talkative and provides lots of specific details from its context. "
    "If the {ai} does not know the answer to a question, it truthfully says it does not know.\n\n"
    "Current conversation:\n\n"
)

ROLE_CONTENT_TMPL_CHAT1 = "### {role}: {content}\n"
