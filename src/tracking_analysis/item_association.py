"""Heuristically link detected people to nearby detected dangerous items."""

from __future__ import annotations

import math


def link_people_to_detected_items(
    people: list[dict],
    items: list[dict],
    *,
    max_distance_person_heights: float = 0.20,
) -> list[dict]:
    """Return only person boxes spatially linked to an item detection.

    Each detected item is linked to its best nearby person box. This is a
    proximity heuristic for candidate tracking; it does not identify who used
    an item or determine who committed violence.
    """
    if not people or not items:
        return []

    linked: dict[int, list[str]] = {}
    for item in items:
        ix1, iy1, ix2, iy2 = map(float, item["bbox"])
        item_center_x = (ix1 + ix2) / 2
        item_center_y = (iy1 + iy2) / 2
        item_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if item_area <= 0:
            continue

        ranked_people = []
        for person_index, person in enumerate(people):
            px1, py1, px2, py2 = map(float, person["bbox"])
            person_height = max(1.0, py2 - py1)
            person_center_x = (px1 + px2) / 2
            person_center_y = (py1 + py2) / 2

            overlap_width = max(0.0, min(px2, ix2) - max(px1, ix1))
            overlap_height = max(0.0, min(py2, iy2) - max(py1, iy1))
            item_overlap = (overlap_width * overlap_height) / item_area
            distance_x = max(px1 - item_center_x, 0.0, item_center_x - px2)
            distance_y = max(py1 - item_center_y, 0.0, item_center_y - py2)
            relative_edge_distance = (
                math.hypot(distance_x, distance_y) / person_height
            )
            relative_center_distance = math.hypot(
                person_center_x - item_center_x, person_center_y - item_center_y
            ) / person_height

            if (
                item_overlap < 0.10
                and relative_edge_distance > max_distance_person_heights
            ):
                continue
            ranked_people.append(
                (
                    (item_overlap, -relative_edge_distance, -relative_center_distance),
                    person_index,
                )
            )

        ranked_people.sort(reverse=True)
        if not ranked_people:
            continue
        if len(ranked_people) > 1:
            best_score, _ = ranked_people[0]
            second_score, _ = ranked_people[1]
            if all(
                abs(best - second) < 0.02
                for best, second in zip(best_score, second_score, strict=True)
            ):
                continue

        person_index = ranked_people[0][1]
        linked.setdefault(person_index, []).append(str(item.get("class_name", "item")))

    return [
        {
            **people[index],
            "associated_items": sorted(set(item_names)),
            "association_type": "item_proximity_candidate",
        }
        for index, item_names in sorted(linked.items())
    ]
