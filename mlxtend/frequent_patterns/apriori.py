# Sebastian Raschka 2014-2019
# myxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd
import itertools
from ..frequent_patterns import fpcommon as fpc


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
    Array of all combinations from the last step x items
    from the previous step.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/generate_new_combinations/

    """

    def generate_pairs(x):
        length = len(x)
        npairs = length * (length - 1) // 2
        pairs = np.fromiter(itertools.chain.from_iterable(itertools.combinations(x, 2)),
                            int, count=2*npairs)
        return pairs.reshape(-1, 2)

    def generate_pairs_np(x):
        # Unfortunately there is no Numpy function to do that, and itertools.combinations
        # is said to be slower than this solution.  In practice there does not seem to
        # be much difference
        length = len(x)
        npairs = length * (length - 1) // 2
        out = np.empty((npairs, 2), dtype=int)
        mask = ~np.tri(length, dtype=bool)
        out[:,0] = np.broadcast_to(x[:, None], (length, length))[mask]
        out[:,1] = np.broadcast_to(x, (length, length))[mask]
        return out

    if old_combinations.shape[-1] == 1:
        # If itemsets are of length 1, generate all pairs.
        return generate_pairs(old_combinations[:, 0])

    # Otherwise, we apply the same algorithm as in apriori-gen
    # Find rows with the same (k-1)-prefix
    rows_diff = np.diff(old_combinations[:, :-1], axis=0)
    index_rows = 1 + np.where(np.any(rows_diff != 0, axis=1))[0]
    change_prefix = np.zeros(2 + len(index_rows), dtype=int)
    change_prefix[1:-1] = index_rows
    change_prefix[-1] = len(old_combinations)
    length_prefix = np.diff(change_prefix)
    size_prefix = length_prefix * (length_prefix - 1) // 2
    k_minus_one = old_combinations.shape[1] - 1
    out = np.empty((size_prefix.sum(), 2 + k_minus_one), dtype=int)
    current_index = 0
    for i in range(len(length_prefix)):
        # all rows from old_combinations between change_prefix[i] and
        # change_prefix[i+1] share the same prefix
        out[current_index:current_index+size_prefix[i], :k_minus_one] = old_combinations[change_prefix[i], :-1]
        out[current_index:current_index+size_prefix[i], k_minus_one:] = generate_pairs(old_combinations[change_prefix[i]:change_prefix[i+1], -1])
        current_index += size_prefix[i]
    return out


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
        X = df.values
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

        # With exceptionally large datasets, the matrix operations can use a
        # substantial amount of memory. For low memory applications or large
        # datasets, set `low_memory=True` to use a slower but more memory-
        # efficient implementation.
        combin = generate_new_combinations(itemset_dict[max_itemset])

        if combin.size == 0:
            break
        if verbose:
            print(
                '\rProcessing %d combinations | Sampling itemset size %d' %
                (combin.size, next_max_itemset), end="")

        itemset_dict_k = []
        support_dict_k = []
        nblocks = 1 + len(combin) // 10 if low_memory else 1
        for sub_combin in np.array_split(combin, nblocks, axis=0):
            if is_sparse:
                _bools = X[:, sub_combin[:, 0]] == all_ones
                for n in range(1, sub_combin.shape[1]):
                    _bools = _bools & (X[:, sub_combin[:, n]] == all_ones)
            else:
                _bools = np.all(X[:, sub_combin], axis=2)

            support = _support(np.array(_bools), rows_count, is_sparse)
            _mask = (support >= min_support).reshape(-1)
            if any(_mask):
                itemset_dict_k.append(sub_combin[_mask])
                support_dict_k.append(support[_mask])

        if itemset_dict_k:
            itemset_dict[next_max_itemset] = np.concatenate(itemset_dict_k)
            support_dict[next_max_itemset] = np.concatenate(support_dict_k)
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
