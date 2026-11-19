"""Admin scope validation happens before opening resources or deleting cache."""
from unittest.mock import Mock

import pytest

from tomte.engine import Tomte
from tomte.engine.clients.admin_client import TomteAdminClient
from tomte.engine.exceptions import TomteInvalidIdError


@pytest.mark.parametrize('method', ['get_user_graph_raw', 'wipe_user_memory'])
@pytest.mark.parametrize('tenant,user', [('tenant:other', 'user'), ('tenant', 'other:user'),
                                         ('tenant', ''), ('', 'user'), ('tenant', None)])
def test_admin_rejects_invalid_scope_before_any_resource_access(monkeypatch, method, tenant, user):
    context, cache = Mock(), Mock()
    monkeypatch.setattr(Tomte, 'operation_context', context)
    monkeypatch.setattr(Tomte, 'get_cache', cache)
    with pytest.raises(TomteInvalidIdError):
        getattr(TomteAdminClient(tenant), method)(user)
    context.assert_not_called()
    cache.assert_not_called()
