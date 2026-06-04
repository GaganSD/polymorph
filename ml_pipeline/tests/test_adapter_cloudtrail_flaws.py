"""Tests for cloudtrail_flaws CSV adapter."""

from pathlib import Path

from polymorph_lamr.distill.adapters import cloudtrail_flaws
from polymorph_lamr.distill.adapters._common import stream_csv_to_txt

FIXTURE_CSV = """\
eventID,eventTime,sourceIPAddress,userAgent,eventName,eventSource,awsRegion,eventVersion,userIdentitytype,eventType,requestID,userIdentityaccountId,userIdentityprincipalId,userIdentityarn,userIdentityaccessKeyId,userIdentityuserName,errorCode,errorMessage,requestParametersinstanceType
abc,2017-02-12T19:57:06Z,10.0.0.1,console,ListBuckets,s3.amazonaws.com,us-east-1,1.05,Root,AwsApiCall,r,811596193553,811596193553,arn:aws:iam::811596193553:root,,,,,
"""

EXPECTED = (
    '2017-02-12T19:57:06Z ListBuckets s3.amazonaws.com region=us-east-1 '
    "identity_type=Root event_type=AwsApiCall arn=arn:aws:iam::811596193553:root "
    'user= src_ip=10.0.0.1 agent="console" principal=811596193553 err= err_msg="" '
    "req_instance_type="
)


def test_render_row_exact_line():
    row = {
        "eventTime": "2017-02-12T19:57:06Z",
        "eventName": "ListBuckets",
        "eventSource": "s3.amazonaws.com",
        "awsRegion": "us-east-1",
        "userIdentitytype": "Root",
        "eventType": "AwsApiCall",
        "userIdentityarn": "arn:aws:iam::811596193553:root",
        "userIdentityuserName": "",
        "sourceIPAddress": "10.0.0.1",
        "userAgent": "console",
        "userIdentityprincipalId": "811596193553",
        "errorCode": "",
        "errorMessage": "",
        "requestParametersinstanceType": "",
    }
    assert cloudtrail_flaws.render_row(row) == EXPECTED


def test_stream_csv_to_txt(tmp_path: Path):
    csv_path = tmp_path / "input.csv"
    csv_path.write_text(FIXTURE_CSV, encoding="utf-8")
    out_path = tmp_path / "out.txt"
    written, skipped = stream_csv_to_txt(
        csv_path,
        out_path,
        render_row=cloudtrail_flaws.render_row,
        required_columns=cloudtrail_flaws._REQUIRED,
    )
    assert written == 1
    assert skipped == 0
    assert out_path.read_text(encoding="utf-8").strip() == EXPECTED
