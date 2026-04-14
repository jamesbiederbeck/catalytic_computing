"""Static analysis passes for Cook-Mertz v2 IR."""

from .cycle_check import check_cycles, check_cycles_diagnostic, find_back_edges
from .field_consistency import check_field_consistency
from .catalytic_verify import verify_catalytic, verify_region
from .cost_propagation import propagate_costs, check_matpow_budget
