"""
Provides a collection of common useful operations on sparse array domains.
"""
from lalcheck.ai.utils import partition


def get(domain):
    """
    :param lalcheck.domains.SparseArray domain: The sparse array domain.

    :return: A function which returns all values that can be get from the
        given indices.

    :rtype: (list, object) -> object
    """
    index_dom = domain.index_dom
    elem_dom = domain.elem_dom

    def do(array, index):
        """
        :param list array: A set of arrays to get from, represented by an
            element of the sparse array domain.

        :param object index: A set of indices to get from, represented by an
            element of the sparse array domain's index domain.

        :return: A set of values that can be get from the given indices, as
            an element of the sparse array domain's element domain.

        :rtype: object
        """
        relevant = [
            elem[1]
            for elem in array
            if not index_dom.is_empty(
                index_dom.meet(index, elem[0])
            )
        ]

        return reduce(elem_dom.join, relevant, elem_dom.bottom)

    return do


def updated(domain):
    """
    :param lalcheck.domains.SparseArray domain: The sparse array domain.

    :return: A function which returns all the arrays that can result after
        updating it to the given index with the given value.

    :rtype: (list, object, object) -> list
    """
    index_dom = domain.index_dom

    def do(array, val, indices):
        """
        :param list array: A set of arrays to update, represented by an
            element of the sparse array domain.

        :param object val: A set of concrete values to update the arrays with,
            represented by an element of the sparse array domain's element
            domain.

        :param object indices: A set of indices to update the array at,
            represented by an element of the sparse array domain's index
            domain.

        :return: A new set of arrays resulting from updating the given arrays
            at the given indices with the given values, represented by an
            element of the sparse array domain.

        :rtype: list
        """

        if index_dom.size(indices) == 1:
            not_relevant, relevant = partition(
                array,
                lambda elem: index_dom.is_empty(
                    index_dom.meet(indices, elem[0])
                )
            )

            updated_relevant = [
                (split, elem[1])
                for elem in relevant
                for split in index_dom.split(elem[0], indices)
                if not index_dom.is_empty(split)
            ]

            return domain.optimized(
                not_relevant +
                updated_relevant +
                [(indices, val)]
            )
        else:
            return domain.join(
                array,
                [(indices, val)]
            )

    return do


def inv_get(domain):
    """
    :param lalcheck.domains.SparseArray domain: The sparse array domain.

    :return: A function which performs the inverse of the get operation.

    :rtype: (object, list, object) -> (list, object)
    """
    index_dom = domain.index_dom
    elem_dom = domain.elem_dom
    do_get = get(domain)
    do_updated = updated(domain)

    def do(res, array_constr, index_constr):
        """
        :param object res: The set of values corresponding to an output of the
            get operation, represented by an element of the sparse array
            domain's element domain.

        :param list array_constr: A constraint on the set of arrays.

        :param object index_constr: A constraint on the set of indices.

        :return: A set of arrays which contain the expected values at the
            given indices, and these indices.
        """
        biggest_array = [
            (split, elem_dom.top)
            for split in index_dom.split(
                index_dom.top,
                index_constr
            )
        ] + [(index_constr, res)]

        array_meet = domain.meet(biggest_array, array_constr)

        if domain.is_empty(array_meet):
            return None

        indices = reduce(index_dom.join, [
            i
            for i, v in array_meet
            if index_dom.le(i, index_constr) and elem_dom.le(v, res)
        ], index_dom.bottom)

        indices_size = index_dom.size(indices)

        if indices_size == 0:
            return None
        elif indices_size == 1:
            return do_updated(
                array_constr,
                elem_dom.meet(res, do_get(array_constr, indices)),
                indices
            ), indices
        else:
            return array_constr, indices

    return do


def inv_updated(domain):
    def do(res, array_constr, val_constr, indices_constr):
        raise NotImplementedError

    return do


def lit(_):
    raise NotImplementedError
