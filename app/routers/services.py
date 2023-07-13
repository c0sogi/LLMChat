import os
from time import sleep
from typing import Any

import boto3
import orjson
import requests
import yagmail
from botocore.exceptions import ClientError
from fastapi import APIRouter
from fastapi.logger import logger
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTasks
from starlette.requests import Request

from app.common.config import (
    AWS_ACCESS_KEY,
    AWS_AUTHORIZED_EMAIL,
    AWS_SECRET_KEY,
    KAKAO_IMAGE_URL,
    KAKAO_RESTAPI_TOKEN,
    WEATHERBIT_API_KEY,
    config,
)
from app.errors.api_exceptions import Responses_400
from app.models.base_models import KakaoMsgBody, MessageOk, SendEmail
from app.utils.api.weather import fetch_weather_data
from app.utils.encoding_utils import encode_from_utf8

router = APIRouter(prefix="/services")
router.redirect_slashes = False


@router.get("/")
async def get_all_services(request: Request):
    return {"your_email": request.state.user.email}


@router.get("/weather", status_code=200)
async def weather(latitude: float, longitude: float):
    if WEATHERBIT_API_KEY is None:
        raise Responses_400.not_supported_feature
    weather_data: Any = await fetch_weather_data(
        lat=latitude,
        lon=longitude,
        api_key=WEATHERBIT_API_KEY,
        source="weatherbit",
    )
    return JSONResponse(
        weather_data,
    )


@router.post("/kakao/send")
async def send_kakao(request: Request, body: KakaoMsgBody):
    link_1 = "https://google.com"
    link_2 = "https://duckduckgo.com"
    headers = {
        "Authorization": KAKAO_RESTAPI_TOKEN,
        "Content-Type": "application/x-www-form-urlencoded",
    }

    template_object = orjson.dumps(
        {
            "object_type": "feed",
            "content": {
                "title": "알림톡",
                "description": body.msg,
                "image_url": KAKAO_IMAGE_URL,
                "link": {
                    "web_url": "http://google.com",
                    "mobile_web_url": "http://google.com",
                    "android_execution_params": "contentId=100",
                    "ios_execution_params": "contentId=100",
                },
            },
            "buttons": [
                {
                    "title": "바로 가기1",
                    "link": {"web_url": link_1, "mobile_web_url": link_1},
                },
                {
                    "title": "바로 가기2",
                    "link": {"web_url": link_2, "mobile_web_url": link_2},
                },
            ],
        },
    )
    data = {"template_object": template_object}
    print(data)
    res = requests.post(
        "https://kapi.kakao.com/v2/api/talk/memo/default/send",
        headers=headers,
        data=data,
    )
    try:
        res.raise_for_status()
        if res.json()["result_code"] != 0:
            raise Exception("KAKAO SEND FAILED")
    except Exception as e:
        logger.warning(e)
        raise Responses_400.kakao_send_failure
    return MessageOk()


email_content = (
    "<div style='margin-top:0cm;margin-right:0cm;margin-bottom:10.0pt;margin-left:0cm"
    ';line-height:115%;font-size:15px;font-family:"Calibri",sans-serif;border:none;bo'
    "rder-bottom:solid #EEEEEE 1.0pt;padding:0cm 0cm 6.0pt 0cm;background:white;'>\n\n<"
    "p style='margin-top:0cm;margin-right:0cm;margin-bottom:11.25pt;margin-left:0cm;l"
    'ine-height:115%;font-size:15px;font-family:"Calibri",sans-serif;background:white'
    ";border:none;padding:0cm;'><span style='font-size:25px;font-family:\"Helvetica Ne"
    "ue\";color:#11171D;'>{}님! Aristoxeni ingenium consumptum videmus in musicis?</spa"
    "n></p>\n</div>\n\n<p style='margin-top:0cm;margin-right:0cm;margin-bottom:11.25pt;m"
    'argin-left:0cm;line-height:17.25pt;font-size:15px;font-family:"Calibri",sans-ser'
    "if;background:white;vertical-align:baseline;'><span style='font-size:14px;font-f"
    'amily:"Helvetica Neue";color:#11171D;\'>Lorem ipsum dolor sit amet, consectetur a'
    "dipiscing elit. Quid nunc honeste dicit? Tum Torquatus: Prorsus, inquit, assenti"
    "or; Duo Reges: constructio interrete. Iam in altera philosophiae parte. Sed haec"
    " omittamus; Haec para/doca illi, nos admirabilia dicamus. Nihil sane.</span></p>"
    "\n\n<p style='margin-top:0cm;margin-right:0cm;margin-bottom:10.0pt;margin-left:0cm"
    ';line-height:normal;font-size:15px;font-family:"Calibri",sans-serif;background:w'
    "hite;'><strong><span style='font-size:24px;font-family:\"Helvetica Neue\";color:#1"
    "1171D;'>Expressa vero in iis aetatibus, quae iam confirmatae sunt.</span></stron"
    "g></p>\n\n<p style='margin-top:0cm;margin-right:0cm;margin-bottom:11.25pt;margin-l"
    'eft:0cm;line-height:17.25pt;font-size:15px;font-family:"Calibri",sans-serif;back'
    "ground:white;vertical-align:baseline;'><span style='font-size:14px;font-family:\""
    "Helvetica Neue\";color:#11171D;'>Sit sane ista voluptas. Non quam nostram quidem,"
    " inquit Pomponius iocans; An tu me de L. Sed haec omittamus; Cave putes quicquam"
    " esse verius.&nbsp;</span></p>\n\n<p style='margin-top:0cm;margin-right:0cm;margin"
    '-bottom:11.25pt;margin-left:0cm;line-height:17.25pt;font-size:15px;font-family:"'
    "Calibri\",sans-serif;text-align:center;background:white;vertical-align:baseline;'"
    "><span style='font-size:14px;font-family:\"Helvetica Neue\";color:#11171D;'><img w"
    'idth="378" src="https://dl1gtqdymozzn.cloudfront.net/forAuthors/K6Sfkx4f2uH780YG'
    'TbEHvHcTX3itiTBtzDWeyswQevxp8jqVttfBgPu86ZtGC6owG.webp" alt="sample1.jpg" class='
    '"fr-fic fr-dii"></span></p>\n\n<p>\n<br>\n</p>'
)


@router.post("/email/send_by_gmail")
async def email_by_gmail(request: Request, mailing_list: SendEmail):
    # t = time()
    send_email(mailing_list=mailing_list.email_to)
    return MessageOk()


@router.post("/email/send_by_gmail2")
async def email_by_gmail2(
    request: Request,
    mailing_list: SendEmail,
    background_tasks: BackgroundTasks,
):
    # t = time()
    background_tasks.add_task(send_email, mailing_list=mailing_list.email_to)
    return MessageOk()


def send_email(**kwargs):
    mailing_list = kwargs.get("mailing_list", None)
    email_password = os.environ.get("EMAIL_PW", None)
    email_addr = os.environ.get("EMAIL_ADDR", None)
    last_email = ""
    if mailing_list:
        try:
            yag = yagmail.SMTP({email_addr: "Admin"}, email_password)
            # https://myaccount.google.com/u/1/lesssecureapps
            for m_l in mailing_list:
                contents = [email_content.format(m_l.name)]
                sleep(1)
                yag.send(m_l.email, "이렇게 한번 보내봅시다.", contents)
                last_email = m_l.email
            return True
        except Exception as e:
            print(e)
            print(last_email)


@router.post("/email/send_by_ses")
async def email_by_ses():
    sender = encode_from_utf8(f"운영자 admin <admin@{config.host_main}>")
    recipient = [AWS_AUTHORIZED_EMAIL]

    # If necessary, replace us-west-2 with the AWS Region you're using for Amazon SES.
    region = "ap-northeast-2"

    # The subject line for the email.
    title = "안녕하세요! 테스트 이메일 입니다."

    # The email body for recipients with non-HTML email clients.
    BODY_TEXT = "안녕하세요! 운영자 입니다.\r\n" "HTML 버전만 지원합니다!"

    # The HTML body of the email.
    BODY_HTML = (
        "<html>\n    <head></head>\n    <body>\n      <h1>안녕하세"
        "요! 반갑습니다.</h1>\n      <p>기업에서 대규모 이메일 솔루션을 구축한다는 것은"
        " 복잡하고 비용이 많이 드는 작업이 될 수 있습니다. 이를 위해서는 인프라를 구축하고, 네"
        "트워크를 구성하고, IP 주소를 준비하고, 발신자 평판을 보호해야 합니다. 타사 이메일 솔"
        "루션 대부분이 상당한 규모의 선수금을 요구하고 계약 협상을 진행해야 합니다.\n\nAmazon"
        " SES는 이러한 부담이 없으므로 몇 분 만에 이메일 발송을 시작할 수 있습니다. Amaz"
        "on.com이 대규모의 자사 고객 기반을 지원하기 위해 구축한 정교한 이메일 인프라와 오랜"
        " 경험을 활용할 수 있습니다.</p>\n      <p>링크를 통해 확인하세요!\n      "
        "  <a href='https://google.com'>Google</a></p>\n    "
        "</body>"
    )

    # The character encoding for the email.
    charset = "UTF-8"

    # Create a new SES resource and specify a region.
    client = boto3.client(
        "ses",
        region_name=region,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )

    # Try to send the email.
    try:
        # Provide the contents of the email.
        response = client.send_email(
            Source=sender,
            Destination={"ToAddresses": recipient},
            Message={
                "Body": {
                    "Html": {"Charset": charset, "Data": BODY_HTML},
                    "Text": {"Charset": charset, "Data": BODY_TEXT},
                },
                "Subject": {"Charset": charset, "Data": title},
            },
        )
    # Display an error if something goes wrong.
    except ClientError as e:
        print(e.response["Error"]["Message"])
    else:
        print("Email sent! Message ID:")
        print(response["MessageId"])

    return MessageOk()
