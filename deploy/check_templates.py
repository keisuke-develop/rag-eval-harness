"""CloudFormation テンプレートのうち、cfn-lint が見ない制約を検査する。

いまは1つだけ。**セキュリティグループのルールの `Description` は ASCII しか通らない。**

cfn-lint は `GroupDescription` は検査するが、`SecurityGroupEgress` /
`SecurityGroupIngress` の中の `Description` は検査しない。
そのため日本語を書くと、cfn-lint と checkov を通り抜けてデプロイ時に落ちる。
実際に1度落としている（deploy/VERIFICATION.md）。

テンプレートの他の `Description`（スタック・パラメータ・出力）は日本語で問題ない。
ここで見るのはセキュリティグループのルールだけ。

    uv run python deploy/check_templates.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml

#: EC2 が受け付ける文字。AWS のエラーメッセージに出てくる集合をそのまま写した。
ALLOWED = re.compile(r"^[A-Za-z0-9. _\-:/()#,@\[\]+=&;{}!$*]*$")
MAX_LENGTH = 255

RULE_KEYS = ("SecurityGroupEgress", "SecurityGroupIngress")


class _Loader(yaml.SafeLoader):
    """`!Ref` や `!Sub` を含むテンプレートを読むためのローダ。

    短縮形のタグは構造を見るぶんには中身が判れば足りるので、
    タグを捨てて値だけを残す。
    """


def _keep_value(loader: _Loader, tag_suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    # yaml の Node はこの3種しかないが、型の上では絞りきれない。
    return None


_Loader.add_multi_constructor("!", _keep_value)


def _rule_descriptions(node: Any) -> list[str]:
    """テンプレートの中から、セキュリティグループのルールの説明だけを集める。"""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in RULE_KEYS and isinstance(value, list):
                for rule in value:
                    if isinstance(rule, dict) and isinstance(rule.get("Description"), str):
                        found.append(rule["Description"])
            found.extend(_rule_descriptions(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_rule_descriptions(item))
    return found


def check(path: Path) -> list[str]:
    template = yaml.load(path.read_text(encoding="utf-8"), Loader=_Loader)  # noqa: S506
    problems = []
    for description in _rule_descriptions(template):
        if not ALLOWED.match(description):
            problems.append(
                f"{path.name}: セキュリティグループのルールの Description に "
                f"使えない文字がある（ASCII のみ）: {description!r}"
            )
        elif len(description) > MAX_LENGTH:
            problems.append(f"{path.name}: ルールの Description が {MAX_LENGTH} 文字を超えている")
    return problems


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv[1:]] or sorted(Path("deploy/cloudformation").glob("*.yaml"))
    problems = [p for path in paths for p in check(path)]
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(
            "\nEC2 が受け付けるのは a-zA-Z0-9 と . _-:/()#,@[]+=&;{}!$* だけ。"
            "\n説明を日本語で残したいなら、YAML のコメントに書くこと。",
            file=sys.stderr,
        )
        return 1
    print(f"セキュリティグループのルールの説明: {len(paths)} ファイル、問題なし")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
