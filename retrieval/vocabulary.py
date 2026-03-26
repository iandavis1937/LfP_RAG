"""
Synonym map and regex helpers for query expansion and token highlighting.
"""

from __future__ import annotations

import re
from itertools import product as _product

SYNONYM_MAP = {
    "armed groups": [
        # actor types
        "armed group","paramilitaries","paramilitary","guerrilla[s|]","insurgent[s|]","combatants",
        "dissident[s|]","FARC","ELN","AGC","bacrim","subersive groups",
        "illegal armed actor[s|]","illegal group[s|]","illegal actor[s|]","factor[s|] illegal",
        "illegal armed group[s|]",
        "group[s|] outside the law","group[s|] on the margin",
        "criminal organization[s|]","criminal group[s|]",
        "armed conflict",
        "grupo[s|] armado[s|]","actor[es|] armado[s|]",
        "actor[es|] ilegal[es|]","grupo[s|] al margen",
        "factor[es|] ilegal[es|]"
    ],

    "violence": [
        # lethal violence
        "assassination[s|]","assassinate[s|d|]","assassinating",
        "kill[s|ed|ing|ings|]","murder[s|ed|ing|]",
        "asesinato[s|]","asesinar","homicidio[s|]",
        # kidnapping
        "kidnap[s|ped|ping|pings|]","abduction[s|]","abduct[s|ed|ing|]",
        "secuestro[s|]","secuestrar",
        # arson
        "arson","set fire","sets fire","setting fire", #"burn[s|ed|ing|]",
        "incendio[s|]","incendiar", #"quemar",
        # looting / theft
        "loot[s|ed|ing|]","pillage[s|d|]","pillaging","theft","robbery",
        "saqueo","saquear","robo","robar",
        # assault / attack
        "assault[s|ed|ing|]","attack[s|ed|ing|]",
        "ataque[s|]","atacar","agresión",
        # gunfire
        "shoot[s|ing|ings|]","shot","shoot-out","gunshot[s|]","gunfire",
        # disappearances
        "disappearance[s|]","disappear[s|ed|ing|]","forced disappearance[s|]",
        "desaparición","desaparecido[s|]","desaparición forzada",
    ],

    "extort": [
        "extortion","extort[s|ed|ing|]","forced payment[s|]","tribute[s|]",
        "vacuna[s|]","cobro[s|]","cuota[s|]","peaje[s|]","pago[s|] forzado[s|]","tributo[s|]",
        "extorsión","extorsionar",
    ],

    "threats": [
        "threat[s|]","threaten[s|ed|ing|]","ultimatum","ultimátum",
        "intimidation","intimidate[s|d|]","intimidating",
        # "scare[s|d|]","scaring","fear[s|ed|ing|]","afraid","paralyze[s|d|]","paralyzing",
        "coerce[s|d|]","coercing","coercion", # "pressure[s|]",
        "amenaza[s|]","amenazar",
    ],

    "restrict access": [
        "checkpoint[s|]","roadblock[s|]","curfew[s|]",
        "restrict[s|ed|] access","access restricted","no-go zone[s|]","control movement",
        "territorial control", #"corridor",
        "retén","retenes","toque de queda","control territorial"
    ],

    "displacement": [
        "displace[s|d|]", "displacement[s|]", "forced displacement",
        "forced migration",
        "abandoned land[s|]",
        "internally displaced",
        "flee[s|d|]", "fleeing", "fled",
        "desplazamiento","desplazado[s|]", "desplazamiento forzado",
        "migración forzada","tierras abandonadas","población desplazada",
        "huy[o|eron|]", "huir",
    ],

    "exploit resources": [
        "exploit[s|ed|ing|] resources","resource extraction",
        "extracción ilegal",
    ],

    "govern": [
        "impose[s|d|] rules","imposing rules","imposed rules",
        "prohibit[s|ed|]", "prohibition[s|]",
        "prohibición[es|]"
    ],

    "alliance": [
        "alliance[s|]","pact[s|]","collude[s|]","collusion",
        "alianza[s|]","pacto[s|]",
    ],

    "deforestation": [
        "[forest|jungle|tree|habitat] loss",
        "[forest|jungle|tree|land] clearing",
        "[forest|jungle|tree|land|habitat] degradation",
        "tala","deforestación",
        ],

    "conservation": [
        "forest protection","protected area[s|]","reserve[s|]","environmental protection",
        "conservación",
        ],

    "government": [
        "governments","state","authorities","authority",
        "municipalities","municipality","municipal",
        "alcaldía","gobernación",
        ],

    "NGO": [
        "organization[s|]","foundation[s|]","civil society","implementing partner[s|]",
        "fundación",
        ],

    "community": [
        "communities","village[s|]","vereda[s|]","resguardo[s|]",
        "resident[s|]","settler[s|]","colono[s|]","campesino[s|]",
        ],

    "illegal economy": [
        "illegal economies","illicit economy","drug trafficking","coca","narcotrafficking",
        "illegal mining","illegal crop[s|]","ilegal","ilícito",
        ],

    "effective": [
        "ineffective","effectively","ineffectively","able","unable",
    ],
}

_BRACKET_ALT_RE = re.compile(r'\[([^\]]+)\]')


def _synonym_to_regex(syn: str) -> str:
    """Convert a synonym string to a regex pattern fragment.

    The [a|b|c] shorthand (e.g. '[forest|jungle] loss') is expanded to a
    non-capturing alternation group '(?:forest|jungle) loss'. Literal portions
    are re.escape()d. Plain strings with no bracket syntax are fully escaped.
    """
    parts = []
    last = 0
    for m in _BRACKET_ALT_RE.finditer(syn):
        parts.append(re.escape(syn[last:m.start()]))
        parts.append('(?:' + '|'.join(re.escape(a) for a in m.group(1).split('|')) + ')')
        last = m.end()
    parts.append(re.escape(syn[last:]))
    return ''.join(parts)


def _expand_synonym(syn: str) -> list[str]:
    """Expand a synonym with [a|b|c] syntax into all plain-string variants.

    '[forest|jungle] loss' → ['forest loss', 'jungle loss'].
    Plain strings are returned as a single-element list.
    """
    if not _BRACKET_ALT_RE.search(syn):
        return [syn]
    segments: list[list[str]] = []
    last = 0
    for m in _BRACKET_ALT_RE.finditer(syn):
        if syn[last:m.start()]:
            segments.append([syn[last:m.start()]])
        segments.append(m.group(1).split('|'))
        last = m.end()
    if syn[last:]:
        segments.append([syn[last:]])
    return [''.join(combo) for combo in _product(*segments)]
