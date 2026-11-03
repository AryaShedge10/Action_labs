"""Draft prescriptions for medications the clinician prescribed or recommended.

Every draft requires a physician signature. Blocked medications are still listed (status
BLOCKED) so the dashboard can show *why* nothing was drafted for them.
"""

from __future__ import annotations

from app.models.schemas import (
    NEW_MEDICATION_STATUSES,
    DraftPrescription,
    EntitySet,
    EntityStatus,
    EntityType,
    PrescriptionStatus,
)
from app.safety.engine import SafetyResult


def build_prescriptions(entities: EntitySet, safety: SafetyResult) -> list[DraftPrescription]:
    drafts: list[DraftPrescription] = []
    for med in entities.of_type(EntityType.MEDICATION):
        if med.status not in NEW_MEDICATION_STATUSES:
            continue
        related = [c for c in safety.conflicts if med.entity_id in c.entity_ids]
        if med.entity_id in safety.blocked_entity_ids:
            status = PrescriptionStatus.BLOCKED
        elif related:
            status = PrescriptionStatus.NEEDS_REVIEW
        else:
            status = PrescriptionStatus.DRAFT

        attrs = med.attributes
        missing = []
        if med.status == EntityStatus.PRESCRIBED:
            missing = [f for f in ("dose", "frequency") if f not in attrs]
            if "duration" not in attrs and "as needed" not in attrs.get("frequency", ""):
                missing.append("duration")
        drafts.append(DraftPrescription(
            prescription_id=f"RX{len(drafts) + 1:03d}",
            entity_id=med.entity_id,
            medication=med.display_name,
            intent=med.status.value,
            dose=attrs.get("dose"),
            frequency=attrs.get("frequency"),
            route=attrs.get("route"),
            duration=attrs.get("duration"),
            status=status,
            missing_fields=missing,
            conflict_ids=[c.conflict_id for c in related],
            evidence=med.evidence,
            confidence=med.confidence,
        ))
    return drafts
