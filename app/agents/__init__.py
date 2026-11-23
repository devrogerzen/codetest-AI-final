from app.agents.analyst_agent import crear_analista
from app.agents.operator_agent import crear_operador
from app.agents.research_agent import crear_investigador
from app.agents.supervisor import crear_supervisor
from app.agents.validator import validador
from app.agents.writer_agent import crear_redactor

__all__ = ["crear_analista", "crear_investigador", "crear_operador", "crear_redactor", "crear_supervisor", "validador"]
