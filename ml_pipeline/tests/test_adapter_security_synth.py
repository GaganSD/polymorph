"""Tests for security_synth multi-vendor CSV adapters."""

from pathlib import Path

from polymorph_lamr.distill.adapters import security_synth
from polymorph_lamr.distill.adapters._common import stream_csv_to_txt

OKTA_CSV = """\
published,severity,eventType,displayMessage,actor.displayName,actor.alternateId,outcome.result,outcome.reason,client.ipAddress,client.geographicalContext.city,client.userAgent.rawUserAgent
2024-01-01T00:00:00Z,INFO,user.session.start,User login,Jane Doe,jane@example.com,SUCCESS,User completed login,203.0.113.1,SF,TestAgent/1.0
"""

OKTA_EXPECTED = (
    '2024-01-01T00:00:00Z INFO okta event=user.session.start msg="User login" '
    "actor=Jane Doe actor_id=jane@example.com result=SUCCESS "
    'reason="User completed login" ip=203.0.113.1 city=SF ua="TestAgent/1.0"'
)

CISCO_CSV = """\
Timestamp,Identities,InternalIp,ExternalIp,Action,QueryType,ResponseCode,Domain,Categories,Verdict,URL
2024-01-01T00:00:00Z,user-1,10.0.0.1,8.8.8.8,Allowed,A,NOERROR,example.com,Search,Benign,http://example.com/
"""

CISCO_EXPECTED = (
    "2024-01-01T00:00:00Z cisco_umbrella identity=user-1 int_ip=10.0.0.1 "
    "ext_ip=8.8.8.8 action=Allowed qtype=A rcode=NOERROR domain=example.com "
    "categories=Search verdict=Benign url=http://example.com/"
)


def test_render_okta_exact_line():
    row = {
        "published": "2024-01-01T00:00:00Z",
        "severity": "INFO",
        "eventType": "user.session.start",
        "displayMessage": "User login",
        "actor.displayName": "Jane Doe",
        "actor.alternateId": "jane@example.com",
        "outcome.result": "SUCCESS",
        "outcome.reason": "User completed login",
        "client.ipAddress": "203.0.113.1",
        "client.geographicalContext.city": "SF",
        "client.userAgent.rawUserAgent": "TestAgent/1.0",
    }
    assert security_synth._render_okta(row) == OKTA_EXPECTED


def test_stream_okta_csv(tmp_path: Path):
    csv_path = tmp_path / "okta.csv"
    csv_path.write_text(OKTA_CSV, encoding="utf-8")
    out_path = tmp_path / "out.txt"
    written, skipped = stream_csv_to_txt(
        csv_path,
        out_path,
        render_row=security_synth._render_okta,
        required_columns=("published", "severity", "eventType"),
    )
    assert written == 1
    assert skipped == 0
    assert out_path.read_text(encoding="utf-8").strip() == OKTA_EXPECTED


def test_render_cisco_dns_exact_line():
    row = {
        "Timestamp": "2024-01-01T00:00:00Z",
        "Identities": "user-1",
        "InternalIp": "10.0.0.1",
        "ExternalIp": "8.8.8.8",
        "Action": "Allowed",
        "QueryType": "A",
        "ResponseCode": "NOERROR",
        "Domain": "example.com",
        "Categories": "Search",
        "Verdict": "Benign",
        "URL": "http://example.com/",
    }
    assert security_synth._render_cisco_dns(row) == CISCO_EXPECTED


def test_stage_corpora_minimal(tmp_path: Path):
    raw = tmp_path / security_synth.SOURCE_DIR
    raw.mkdir(parents=True)
    for csv_name, content in [
        ("okta_system_log.csv", OKTA_CSV),
        ("cisco_umbrella_dns.csv", CISCO_CSV),
        (
            "crowdstrike_network.csv",
            "timestamp,EventType,ConnectionDirection,LocalAddressIP4,LocalPort,RemoteAddressIP4,RemotePort,Protocol,Status,ImageFileName,CommandLine,UserName,ComputerName,Technique\n"
            "2024-01-01T00:00:00Z,NetworkConnect,Outbound,10.0.0.1,443,8.8.8.8,443,TCP,Success,proc.exe,cmd,user1,host1,T1059\n",
        ),
        (
            "crowdstrike_process.csv",
            "ProcessStartTime,EventType,ImageFileName,CommandLine,ParentBaseFileName,UserName,ComputerName,Technique\n"
            "2024-01-01T00:00:00Z,ProcessRollup2,proc.exe,cmd,parent.exe,user1,host1,T1059\n",
        ),
        (
            "crowdstrike_registry.csv",
            "timestamp,EventType,RegObjectName,RegValueName,RegStringValue,RegOperationType,ImageFileName,UserName,ComputerName,Technique\n"
            "2024-01-01T00:00:00Z,RegValueSet,HKLM\\Software,ValueName,val,Set,proc.exe,user1,host1,T1112\n",
        ),
        (
            "proofpoint_email.csv",
            "time,sender,recipient,subject,headerFrom,spamScore,phishScore,mlxLabel,action,malwareName,sendingIp\n"
            '2024-01-01T00:00:00Z,a@x.com,b@y.com,Hello,a@x.com,0,0,clean,Delivered,,1.2.3.4\n',
        ),
        (
            "zscaler_proxy.csv",
            "timestamp,user,department,location,clientip,serverip,hostname,url,urlcategory,action,requestmethod,responsecode,contenttype,useragent,threatname,rulelabel,appname\n"
            "2024-01-01T00:00:00Z,u1,eng,US,10.0.0.1,10.0.0.2,host1,http://x/,News,Allow,GET,200,text/html,UA,,rule1,app1\n",
        ),
        (
            "aws_cloudtrail.csv",
            "eventTime,eventName,eventSource,awsRegion,sourceIPAddress,userAgent,userIdentity.type,userIdentity.principalId,userIdentity.arn,userIdentity.userName,errorCode,requestParameters.bucketName\n"
            "2024-01-01T00:00:00Z,GetObject,s3.amazonaws.com,us-east-1,10.0.0.1,agent,AssumedRole,ARO123,arn:aws:sts::1:assumed-role/r,s,,my-bucket\n",
        ),
    ]:
        (raw / csv_name).write_text(content, encoding="utf-8")

    results = security_synth.stage_corpora(tmp_path)
    assert len(results) == 8
    assert all((tmp_path / r["staged_path"]).exists() for r in results)
    okta_out = tmp_path / "data/staged/okta_auth.txt"
    assert okta_out.read_text(encoding="utf-8").strip() == OKTA_EXPECTED
