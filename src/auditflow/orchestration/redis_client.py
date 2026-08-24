import os
from redis.sentinel import Sentinel

SENTINEL_HOSTS = [
    (h.split(":")[0], int(h.split(":")[1]))
    for h in os.environ["SENTINEL_HOSTS"].split(",")
]
SERVICE_NAME = os.environ.get("REDIS_SERVICE_NAME", "auditflow-redis")

_sentinel = Sentinel(SENTINEL_HOSTS, socket_timeout=0.5)

def get_primary(db: int = 0):
    return _sentinel.master_for(SERVICE_NAME, socket_timeout=0.5, db=db)


def get_replica(db: int = 0):
    return _sentinel.slave_for(SERVICE_NAME, socket_timeout=0.5, db=db)