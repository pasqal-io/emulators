import torch




def permute_bitstrings(results: Results, perm: torch.Tensor) -> None:
    if "bitstrings" not in results.get_result_tags():
        return
    uuid_bs = results._find_uuid("bitstrings")

    results._results[uuid_bs] = [
        Counter({optimat.permute_string(bstr, perm): c for bstr, c in bs_counter.items()})
        for bs_counter in results._results[uuid_bs]
    ]


def permute_occupations_and_correlations(results: Results, perm: torch.Tensor) -> None:
    for corr in ["occupation", "correlation_matrix"]:
        if corr not in results.get_result_tags():
            continue

        uuid_corr = results._find_uuid(corr)
        corrs = results._results[uuid_corr]
        results._results[uuid_corr] = (
            [  # vector quantities become lists after results are serialized (e.g. for checkpoints)
                optimat.permute_tensor(
                    corr if isinstance(corr, torch.Tensor) else torch.tensor(corr), perm
                )
                for corr in corrs
            ]
        )


def permute_atom_order(results: Results, perm: torch.Tensor) -> None:
    at_ord = list(results.atom_order)
    at_ord = optimat.permute_list(at_ord, perm)
    results.atom_order = tuple(at_ord)
