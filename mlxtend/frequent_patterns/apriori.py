# Sebastian Raschka 2014-2019
# myxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd
import pygtrie
from ..frequent_patterns import fpcommon as fpc


def compute_supports(X, is_sparse, combin):
    supports = np.zeros(combin.shape[0])
    ncomb, nitems = combin.shape
    if is_sparse:
        count = np.empty(X.shape[0], dtype=int)
        for c in range(ncomb):
            count[:] = 0
            for j in combin[c]:
                # much faster than X.getcol(j).indices
                count[X.indices[X.indptr[j]:X.indptr[j+1]]] += 1
            supports[c] = np.count_nonzero(count == nitems)
    else:
        _bools = np.copy(X[:, 0])
        for c in range(ncomb):
            _bools[:] = X[:, combin[c, 0]]
            for j in range(1, nitems):
                _bools[:] &= X[:, combin[c, j]]
            supports[c] = np.count_nonzero(_bools)
    return supports


def generate_new_combinations(old_combinations):
    """
    Generator of all combinations based on the last state of Apriori algorithm
    Parameters
    -----------
    old_combinations: np.array
        All combinations with enough support in the last step
        Combinations are represented by a matrix.
        Number of columns is equal to the combination size
        of the previous step.
        Each row represents one combination
        and contains item type ids in the ascending order
        ```
               0        1
        0      15       20
        1      15       22
        2      17       19
        ```

    Returns
    -----------
    Generator of combinations based on the last state of Apriori algorithm.
    In order to reduce number of candidates, this function implements the
    join step of apriori-gen described in section 2.1.1 of Apriori paper.
    Prune step is not implemented because it is too slow in pure Python.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/generate_new_combinations/

    """

    length = len(old_combinations)
    trie = pygtrie.Trie(list(zip(old_combinations, [1]*length)))
    for i, old_combination in enumerate(old_combinations):
        *head_i, _ = old_combination
        j = i + 1
        while j < length:
            *head_j, tail_j = old_combinations[j]
            if head_i != head_j:
                break
            # Prune old_combination+(item,) if any subset is not frequent
            candidate = tuple(old_combination) + (tail_j,)
            for idx in range(len(candidate)):
                test_candidate = list(candidate)
                del test_candidate[idx]
                if tuple(test_candidate) not in trie:
                    # early exit from for-loop skips else clause just below
                    break
            else:
                yield from candidate
            j = j + 1


def apriori(df, min_support=0.5, use_colnames=False, max_len=None, verbose=0,
            low_memory=False):
    """Get frequent itemsets from a one-hot DataFrame

    Parameters
    -----------
    df : pandas DataFrame or pandas SparseDataFrame
      pandas DataFrame the encoded format.
      The allowed values are either 0/1 or True/False.
      For example,

    ```
             Apple  Bananas   Beer  Chicken   Milk   Rice
        0     True    False   True     True  False   True
        1     True    False   True    False  False   True
        2     True    False   True    False  False  False
        3     True     True  False    False  False  False
        4    False    False   True     True   True   True
        5    False    False   True    False   True   True
        6    False    False   True    False   True  False
        7     True     True  False    False  False  False
    ```

    min_support : float (default: 0.5)
      A float between 0 and 1 for minumum support of the itemsets returned.
      The support is computed as the fraction
      `transactions_where_item(s)_occur / total_transactions`.

    use_colnames : bool (default: False)
      If `True`, uses the DataFrames' column names in the returned DataFrame
      instead of column indices.

    max_len : int (default: None)
      Maximum length of the itemsets generated. If `None` (default) all
      possible itemsets lengths (under the apriori condition) are evaluated.

    verbose : int (default: 0)
      Shows the number of iterations if >= 1 and `low_memory` is `True`. If
      >=1 and `low_memory` is `False`, shows the number of combinations.

    low_memory : bool (default: False)
      If `True`, uses an iterator to search for combinations above
      `min_support`.
      Note that while `low_memory=True` should only be used for large dataset
      if memory resources are limited, because this implementation is approx.
      3-6x slower than the default.


    Returns
    -----------
    pandas DataFrame with columns ['support', 'itemsets'] of all itemsets
      that are >= `min_support` and < than `max_len`
      (if `max_len` is not None).
      Each itemset in the 'itemsets' column is of type `frozenset`,
      which is a Python built-in type that behaves similarly to
      sets except that it is immutable
      (For more info, see
      https://docs.python.org/3.6/library/stdtypes.html#frozenset).

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/

    """

    def _support(_x, _n_rows, _is_sparse):
        """DRY private method to calculate support as the
        row-wise sum of values / number of rows

        Parameters
        -----------

        _x : matrix of bools or binary

        _n_rows : numeric, number of rows in _x

        _is_sparse : bool True if _x is sparse

        Returns
        -----------
        np.array, shape = (n_rows, )

        Examples
        -----------
        For usage examples, please see
        http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/

        """
        out = (np.sum(_x, axis=0) / _n_rows)
        return np.array(out).reshape(-1)

    if min_support <= 0.:
        raise ValueError('`min_support` must be a positive '
                         'number within the interval `(0, 1]`. '
                         'Got %s.' % min_support)

    fpc.valid_input_check(df)

    # sparse attribute exists for both deprecated SparseDataFrame and
    # DataFrame with SparseArray (pandas >= 0.24); to_coo attribute
    # exists only for the former, thus it is checked first to distinguish
    # between SparseDataFrame and DataFrame with SparseArray.
    if hasattr(df, "to_coo"):
        # SparseDataFrame with pandas < 0.24
        if df.size == 0:
            X = df.values
        else:
            X = df.to_coo().tocsc()
        is_sparse = True
    elif hasattr(df, "sparse"):
        # DataFrame with SparseArray (pandas >= 0.24)
        if df.size == 0:
            X = df.values
        else:
            X = df.sparse.to_coo().tocsc()
        is_sparse = True
    else:
        # dense DataFrame
        X = np.asfortranarray(df.values)
        is_sparse = False
    support = _support(X, X.shape[0], is_sparse)
    ary_col_idx = np.arange(X.shape[1])
    support_dict = {1: support[support >= min_support]}
    itemset_dict = {1: ary_col_idx[support >= min_support].reshape(-1, 1)}
    max_itemset = 1
    rows_count = float(X.shape[0])

    all_ones = np.ones((int(rows_count), 1))

    while max_itemset and max_itemset < (max_len or float('inf')):
        next_max_itemset = max_itemset + 1

        combin = generate_new_combinations(itemset_dict[max_itemset])
        combin = np.fromiter(combin, dtype=int)
        combin = combin.reshape(-1, next_max_itemset)

        if combin.size == 0:
            break
        if verbose:
            print(
                '\rProcessing %d combinations | Sampling itemset size %d' %
                (combin.size, next_max_itemset), end="")

        support = compute_supports(X, is_sparse, combin) / rows_count
        _mask = (support >= min_support)
        if np.any(_mask):
            itemset_dict[next_max_itemset] = np.array(combin[_mask])
            support_dict[next_max_itemset] = np.array(support[_mask])
            max_itemset = next_max_itemset
        else:
            # Exit condition
            break

    all_res = []
    for k in sorted(itemset_dict):
        support = pd.Series(support_dict[k])
        itemsets = pd.Series([frozenset(i) for i in itemset_dict[k]])

        res = pd.concat((support, itemsets), axis=1)
        all_res.append(res)

    res_df = pd.concat(all_res)
    res_df.columns = ['support', 'itemsets']
    if use_colnames:
        mapping = {idx: item for idx, item in enumerate(df.columns)}
        res_df['itemsets'] = res_df['itemsets'].apply(lambda x: frozenset([
                                                      mapping[i] for i in x]))
    res_df = res_df.reset_index(drop=True)

    if verbose:
        print()  # adds newline if verbose counter was used

    return res_df
