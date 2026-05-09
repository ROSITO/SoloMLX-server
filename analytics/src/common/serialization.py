import json
from datetime import date, datetime


def to_json(value):
    return json.dumps(value, default=_default_serializer, ensure_ascii=False)


def _default_serializer(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type not serializable: {type(obj)}")
