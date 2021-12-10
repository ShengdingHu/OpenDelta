from typing import List
def superstring_in(str_a: str , list_b: List[str]):
    r"""check whether there is any string in list b containing str_a.

    Args:
    Returns:
    """
    return any(str_a in str_b for str_b in list_b)

def substring_in(str_a: str , list_b: List[str]):
    r"""check whether `str_a` has a substring that is in list_b.

    Args:
    Returns:
    """
    return any(str_b in str_a for str_b in list_b)