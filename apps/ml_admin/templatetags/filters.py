# Author: Marcus Berggren
from django import template

register = template.Library()


@register.filter
def percentage(value):
    if value is None:
        return '-'
    return f'{value * 100:.2f}%'
