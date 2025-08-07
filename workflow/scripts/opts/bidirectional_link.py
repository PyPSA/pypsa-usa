import logging  # noqa: D100

logger = logging.getLogger(__name__)


def add_bidirectional_link_constraints(n):
    """
    Add constraints for bidirectional links (transmission and H2 pipelines).

    For pairs of extendable links with identical names except for 'fwd' and 'rev':
    Add constraint: fwd.p_nom_opt - fwd.p_nom = rev.p_nom_opt - rev.p_nom

    This ensures the two links model the same physical infrastructure.
    """

    # Get all extendable links
    extendable_links = n.links[n.links.p_nom_extendable].copy()

    # Find potential bidirectional link pairs
    # These are links that contain either '_fwd' or '_rev' at the end of their names
    bidirectional_candidates = extendable_links[
        extendable_links.index.str.contains(r'_fwd$|_rev$', regex=True, case=True)
    ]

    if bidirectional_candidates.empty:
        logger.info("No bidirectional link candidates found (no _fwd or _rev at the end of the names)")
        return

    # Group links by their base name (removing _fwd or _rev parts)
    link_pairs = {}

    for link_name in bidirectional_candidates.index:
        if '_fwd' in link_name:
            base_name = link_name.replace('_fwd', '', 1)
            if base_name not in link_pairs:
                link_pairs[base_name] = {}
            link_pairs[base_name]['fwd'] = link_name
        elif '_rev' in link_name:
            base_name = link_name.replace('_rev', '', 1)
            if base_name not in link_pairs:
                link_pairs[base_name] = {}
            link_pairs[base_name]['rev'] = link_name

    # Filter to only complete pairs (both fwd and rev exist)
    complete_pairs = {
        base_name: pair for base_name, pair in link_pairs.items()
        if 'fwd' in pair and 'rev' in pair
    }

    if not complete_pairs:
        logger.info("No complete bidirectional link pairs found")
        # Log the incomplete pairs for infoging
        incomplete_pairs = {k: v for k, v in link_pairs.items() if len(v) == 1}
        if incomplete_pairs:
            logger.info(f"Found {len(incomplete_pairs)} incomplete pairs:")
            for base_name, pair in incomplete_pairs.items():
                direction = list(pair.keys())[0]
                link_name = list(pair.values())[0]
                logger.info(f"  {base_name}: only {direction} link ({link_name})")
        return

    constraints_added = 0

    for base_name, pair in complete_pairs.items():
        fwd_link = pair['fwd']
        rev_link = pair['rev']

        # Get link properties
        fwd_p_nom = n.links.loc[fwd_link, 'p_nom']
        rev_p_nom = n.links.loc[rev_link, 'p_nom']

        # Get optimization variables
        fwd_p_nom_opt = n.model["Link-p_nom"].loc[fwd_link]
        rev_p_nom_opt = n.model["Link-p_nom"].loc[rev_link]

        # Add constraint: fwd.p_nom_opt - fwd.p_nom = rev.p_nom_opt - rev.p_nom
        constraint_name = f"bidirectional_link_{base_name.replace(' ', '_').replace('-', '_')}"
        lhs = fwd_p_nom_opt + rev_p_nom - fwd_p_nom - rev_p_nom_opt

        n.model.add_constraints(
            lhs == 0,
            name=constraint_name
        )
        constraints_added += 1

    logger.info(f"Added {constraints_added} bidirectional link constraints")
