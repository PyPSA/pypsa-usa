import logging  # noqa: D100

from constants import FWD_SUFFIX, REV_SUFFIX

logger = logging.getLogger(__name__)


def add_bidirectional_link_constraints(n):
    """
    Add constraints for bidirectional links (transmission and H2 pipelines).

    For pairs of extendable links with identical names except for 'fwd' and 'rev':
    Add constraint: fwd.p_nom_opt - fwd.p_nom = rev.p_nom_opt - rev.p_nom

    Vintaged links ('<name>_fwd_2040' / '<name>_rev_2040') are paired within
    their own vintage.

    This ensures the two links model the same physical infrastructure.
    """
    extendable_links = n.links.index[n.links.p_nom_extendable]

    # Pair each '_fwd' link with the '_rev' link of the same name and vintage
    complete_pairs = {}
    for fwd_link in extendable_links[extendable_links.str.contains(FWD_SUFFIX, regex=False)]:
        rev_link = REV_SUFFIX.join(fwd_link.rsplit(FWD_SUFFIX, 1))
        if rev_link in extendable_links:
            complete_pairs[fwd_link] = rev_link

    if not complete_pairs:
        logger.info("No complete bidirectional link pairs found")
        return

    constraints_added = 0

    for fwd_link, rev_link in complete_pairs.items():
        base_name = "".join(fwd_link.rsplit(FWD_SUFFIX, 1))

        # Get link properties
        fwd_p_nom = n.links.loc[fwd_link, "p_nom"]
        rev_p_nom = n.links.loc[rev_link, "p_nom"]

        # Get optimization variables
        fwd_p_nom_opt = n.model["Link-p_nom"].loc[fwd_link]
        rev_p_nom_opt = n.model["Link-p_nom"].loc[rev_link]

        # Add constraint: fwd.p_nom_opt - fwd.p_nom = rev.p_nom_opt - rev.p_nom
        constraint_name = f"bidirectional_link_{base_name.replace(' ', '_').replace('-', '_')}"
        lhs = fwd_p_nom_opt + rev_p_nom - fwd_p_nom - rev_p_nom_opt

        n.model.add_constraints(
            lhs == 0,
            name=constraint_name,
        )
        constraints_added += 1

    logger.info(f"Added {constraints_added} bidirectional link constraints")
