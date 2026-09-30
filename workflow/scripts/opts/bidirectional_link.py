import logging  # noqa: D100
import re

logger = logging.getLogger(__name__)

# `<name>_fwd` / `<name>_rev`, optionally followed by a vintage: `<name>_fwd_2040`
DIRECTION_SUFFIX = re.compile(r"^(?P<base>.*)_(?P<direction>fwd|rev)(?P<vintage>_\d+)?$")


def add_bidirectional_link_constraints(n):
    """
    Add constraints for bidirectional links (transmission and H2 pipelines).

    For pairs of extendable links with identical names except for 'fwd' and 'rev':
    Add constraint: fwd.p_nom_opt - fwd.p_nom = rev.p_nom_opt - rev.p_nom

    Vintaged links ('<name>_fwd_2040' / '<name>_rev_2040') are paired within
    their own vintage.

    This ensures the two links model the same physical infrastructure.
    """
    # Get all extendable links
    extendable_links = n.links[n.links.p_nom_extendable].copy()

    # Find potential bidirectional link pairs
    # These are links whose names end in '_fwd' or '_rev', with an optional '_<vintage>' after it
    matches = {link_name: DIRECTION_SUFFIX.match(link_name) for link_name in extendable_links.index}
    bidirectional_candidates = {link_name: m for link_name, m in matches.items() if m}

    if not bidirectional_candidates:
        logger.info("No bidirectional link candidates found (no _fwd or _rev at the end of the names)")
        return

    # Group links by their base name (removing the _fwd or _rev part, keeping the vintage)
    link_pairs = {}

    for link_name, m in bidirectional_candidates.items():
        base_name = m["base"] + (m["vintage"] or "")
        link_pairs.setdefault(base_name, {})[m["direction"]] = link_name

    # Filter to only complete pairs (both fwd and rev exist)
    complete_pairs = {base_name: pair for base_name, pair in link_pairs.items() if "fwd" in pair and "rev" in pair}

    if not complete_pairs:
        logger.info("No complete bidirectional link pairs found")
        # Log the incomplete pairs for infoging
        incomplete_pairs = {k: v for k, v in link_pairs.items() if len(v) == 1}
        if incomplete_pairs:
            logger.info(f"Found {len(incomplete_pairs)} incomplete pairs:")
            for base_name, pair in incomplete_pairs.items():
                direction = next(iter(pair.keys()))
                link_name = next(iter(pair.values()))
                logger.info(f"  {base_name}: only {direction} link ({link_name})")
        return

    constraints_added = 0

    for base_name, pair in complete_pairs.items():
        fwd_link = pair["fwd"]
        rev_link = pair["rev"]

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
