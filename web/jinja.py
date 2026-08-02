"""One shared Jinja2Templates instance — every router renders through this."""

from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="web/templates")
