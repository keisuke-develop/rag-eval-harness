"""OS の証明書ストアを PEM に書き出す（TLS 傍受のある環境向け）。

社内プロキシなどで TLS が傍受される環境だと、AWS CLI が
`SSL: CERTIFICATE_VERIFY_FAILED` で接続できない。このスクリプトで
OS 側の証明書を書き出し、`AWS_CA_BUNDLE` に渡す。

    python deploy/make_ca_bundle.py ca-bundle.pem
    export AWS_CA_BUNDLE=$PWD/ca-bundle.pem

`--no-verify-ssl` で黙らせないこと。証明書の検証を切ると、
傍受されていることと攻撃されていることを区別できなくなる。

Windows 以外では `ssl.enum_certificates` が無いので、その環境では
OS のバンドル（Debian 系なら /etc/ssl/certs/ca-certificates.crt）を直接指定する。
"""

from __future__ import annotations

import base64
import ssl
import sys
from pathlib import Path


def collect() -> list[str]:
    if not hasattr(ssl, "enum_certificates"):
        raise SystemExit(
            "この OS には ssl.enum_certificates が無い。"
            "OS のバンドル（例: /etc/ssl/certs/ca-certificates.crt）を"
            "直接 AWS_CA_BUNDLE に指定する。"
        )
    pems: list[str] = []
    for store in ("ROOT", "CA"):
        for der, encoding, _trust in ssl.enum_certificates(store):
            if encoding != "x509_asn":
                continue
            body = base64.encodebytes(der).decode("ascii")
            pems.append(f"-----BEGIN CERTIFICATE-----\n{body}-----END CERTIFICATE-----\n")
    return pems


def main(argv: list[str]) -> int:
    out = Path(argv[1] if len(argv) > 1 else "ca-bundle.pem")
    pems = collect()
    out.write_text("".join(pems), encoding="ascii", newline="\n")
    print(f"{len(pems)} 証明書を {out} に書き出した。")
    print(f"export AWS_CA_BUNDLE={out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
