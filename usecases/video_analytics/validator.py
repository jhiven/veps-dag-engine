"""Validator for real-world reconfiguration experiment result CSV files."""

from __future__ import annotations

import csv
import os

class ValidationFailure(Exception):
    """Exception raised when a validation invariant is violated."""
    pass

def validate_benchmark_csvs(summary_csv_path: str, frame_csv_path: str) -> tuple[bool, list[str]]:
    """Validate summary and frame CSV files against methodological invariants."""
    errors: list[str] = []

    if not os.path.exists(summary_csv_path):
        errors.append(f"Summary CSV file not found: {summary_csv_path}")
        return False, errors
    if not os.path.exists(frame_csv_path):
        errors.append(f"Frame CSV file not found: {frame_csv_path}")
        return False, errors

    # Load summary records
    summary_rows: list[dict[str, str]] = []
    with open(summary_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            summary_rows.append(row)

    # Load frame records
    frame_rows: list[dict[str, str]] = []
    with open(frame_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_rows.append(row)

    def parse_int(val: str | None) -> int | None:
        if not val or val.strip() == "":
            return None
        try:
            return int(val)
        except ValueError:
            return None

    # Group frame rows by (repetition, mechanism)
    frames_by_run: dict[tuple[int, str], list[dict[str, str]]] = {}
    for fr in frame_rows:
        rep = parse_int(fr.get("repetition"))
        mech = fr.get("mechanism")
        if rep is not None and mech:
            key = (rep, mech)
            frames_by_run.setdefault(key, []).append(fr)

    # 1. Per-frame validation
    for key, f_list in frames_by_run.items():
        rep, mech = key
        seen_ids: set[int] = set()
        tracker_ids: set[str] = set()
        
        for fr in f_list:
            fid = parse_int(fr.get("frame_id"))
            if fid is None:
                errors.append(f"Rep {rep} Mech {mech}: Frame record lacks valid frame_id")
                continue

            # Unique frame identity
            if fid in seen_ids:
                errors.append(f"Rep {rep} Mech {mech} Frame {fid}: Duplicate frame_id in run")
            seen_ids.add(fid)

            # Parse timestamps
            t_ingress = parse_int(fr.get("receiver_ingress_timestamp_ns"))
            t_enqueue = parse_int(fr.get("enqueue_decision_timestamp_ns"))
            t_admit = parse_int(fr.get("admission_timestamp_ns"))
            t_comp = parse_int(fr.get("completion_timestamp_ns"))
            t_drop = parse_int(fr.get("drop_decision_timestamp_ns"))

            dropped = parse_int(fr.get("dropped")) == 1 or fr.get("dropped") == "True"
            duplicated = parse_int(fr.get("duplicated")) == 1 or fr.get("duplicated") == "True"
            term_status = fr.get("terminal_status", "")
            drop_reason = fr.get("drop_reason", "")
            inside = parse_int(fr.get("inside_measurement_window")) == 1 or fr.get("inside_measurement_window") == "True"

            # Check impossible terminal state combinations
            if dropped and duplicated:
                errors.append(f"Rep {rep} Mech {mech} Frame {fid}: Frame cannot be both dropped and duplicated")

            if term_status == "completed":
                if dropped:
                    errors.append(f"Rep {rep} Mech {mech} Frame {fid}: Completed frame has dropped=True")
                if drop_reason != "none" and drop_reason != "":
                    errors.append(f"Rep {rep} Mech {mech} Frame {fid}: Completed frame has drop_reason {drop_reason}")
                if t_admit is None or t_comp is None:
                    errors.append(f"Rep {rep} Mech {mech} Frame {fid}: Completed frame lacks admission or completion timestamp")

            if dropped:
                if t_comp is not None:
                    errors.append(f"Rep {rep} Mech {mech} Frame {fid}: Dropped frame has completion timestamp")
                if drop_reason == "none" or not drop_reason:
                    errors.append(f"Rep {rep} Mech {mech} Frame {fid}: Dropped frame lacks valid drop_reason")
                if t_drop is None:
                    errors.append(f"Rep {rep} Mech {mech} Frame {fid}: Dropped frame lacks drop_decision_timestamp_ns")

            # Causal timestamp ordering
            if t_ingress is not None and t_enqueue is not None:
                if t_ingress > t_enqueue:
                    errors.append(f"Rep {rep} Mech {mech} Frame {fid}: Ingress timestamp ({t_ingress}) > enqueue timestamp ({t_enqueue})")

            if not dropped:
                if t_enqueue is not None and t_admit is not None:
                    if t_enqueue > t_admit:
                        errors.append(f"Rep {rep} Mech {mech} Frame {fid}: Enqueue timestamp ({t_enqueue}) > admission timestamp ({t_admit})")
                if t_admit is not None and t_comp is not None:
                    if t_admit > t_comp:
                        errors.append(f"Rep {rep} Mech {mech} Frame {fid}: Admission timestamp ({t_admit}) > completion timestamp ({t_comp})")
            else:
                if t_enqueue is not None and t_drop is not None:
                    if t_enqueue > t_drop:
                        errors.append(f"Rep {rep} Mech {mech} Frame {fid}: Enqueue timestamp ({t_enqueue}) > drop timestamp ({t_drop})")

            # Queue occupancy range check
            cap = parse_int(fr.get("queue_capacity"))
            occ_bef = parse_int(fr.get("queue_occupancy_before_enqueue"))
            occ_aft = parse_int(fr.get("queue_occupancy_after_enqueue"))
            if cap is not None:
                if occ_bef is not None and not (0 <= occ_bef <= cap):
                    errors.append(f"Rep {rep} Mech {mech} Frame {fid}: Occupancy before ({occ_bef}) out of bounds [0, {cap}]")
                if occ_aft is not None and not (0 <= occ_aft <= cap):
                    errors.append(f"Rep {rep} Mech {mech} Frame {fid}: Occupancy after ({occ_aft}) out of bounds [0, {cap}]")

            # Plan version / detector agreement
            p_ver = parse_int(fr.get("plan_version"))
            det_id = fr.get("detector_id")
            if p_ver is not None and det_id:
                if p_ver == 1 and "rtdetr_r18" not in det_id.lower() and "r18" not in det_id.lower():
                    errors.append(f"Rep {rep} Mech {mech} Frame {fid}: Plan version 1 has incompatible detector_id {det_id}")
                elif p_ver == 2 and "rtdetr_r50" not in det_id.lower() and "r50" not in det_id.lower():
                    errors.append(f"Rep {rep} Mech {mech} Frame {fid}: Plan version 2 has incompatible detector_id {det_id}")

            # Tracker identity collection for PRESERVE check
            trk_id = fr.get("tracker_instance_id")
            if trk_id and inside:
                tracker_ids.add(trk_id)

        # Tracker preserve check
        if len(tracker_ids) > 1:
            errors.append(f"Rep {rep} Mech {mech}: Tracker identity not preserved across swaps (found multiple: {tracker_ids})")

    # 2. Per-run validation
    for s in summary_rows:
        rep = parse_int(s.get("repetition"))
        mech = s.get("mechanism")
        if rep is None or not mech:
            continue

        run_frames = frames_by_run.get((rep, mech), [])
        
        # Denominators and Phase counts
        pre_req_target = parse_int(s.get("pre_request_source_frame_target")) or 60
        pre_req_rec = parse_int(s.get("pre_request_source_frames_received"))
        post_eff_target = parse_int(s.get("post_effect_source_frame_target")) or 60
        post_eff_rec = parse_int(s.get("post_effect_source_frames_received"))
        total_rec = parse_int(s.get("total_measurement_source_frames_received"))

        if pre_req_rec is not None and pre_req_rec != pre_req_target:
            errors.append(f"Rep {rep} Mech {mech}: pre-request received ({pre_req_rec}) != target ({pre_req_target})")
        if post_eff_rec is not None and post_eff_rec != post_eff_target:
            errors.append(f"Rep {rep} Mech {mech}: post-effect received ({post_eff_rec}) != target ({post_eff_target})")

        # Phase source and drop counts reconciliation
        s_before = parse_int(s.get("source_frames_before_request"))
        s_prep = parse_int(s.get("source_frames_during_candidate_preparation"))
        s_prep_pub = parse_int(s.get("source_frames_between_preparation_and_publication"))
        s_pub_first = parse_int(s.get("source_frames_between_publication_and_first_candidate_output"))
        s_after = parse_int(s.get("source_frames_after_first_candidate_output"))

        d_before = parse_int(s.get("drops_before_request"))
        d_prep = parse_int(s.get("drops_during_candidate_preparation"))
        d_prep_pub = parse_int(s.get("drops_between_preparation_and_publication"))
        d_pub_first = parse_int(s.get("drops_between_publication_and_first_candidate_output"))
        d_after = parse_int(s.get("drops_after_first_candidate_output"))

        total_drops = parse_int(s.get("total_dropped_frame_count"))

        if total_rec is not None:
            sum_src = sum(x for x in (s_before, s_prep, s_prep_pub, s_pub_first, s_after) if x is not None)
            if sum_src != total_rec:
                errors.append(f"Rep {rep} Mech {mech}: Sum of phase source frames ({sum_src}) != total received ({total_rec})")

        if total_drops is not None:
            sum_drp = sum(x for x in (d_before, d_prep, d_prep_pub, d_pub_first, d_after) if x is not None)
            if sum_drp != total_drops:
                errors.append(f"Rep {rep} Mech {mech}: Sum of phase drops ({sum_drp}) != total dropped ({total_drops})")

        # Accounting reconciliation
        completed = parse_int(s.get("frames_completed"))
        overflow = parse_int(s.get("ingress_overflow_drop_count"))
        rejected = parse_int(s.get("admission_rejection_count"))
        cancelled = parse_int(s.get("execution_cancelled_count"))
        in_flight = parse_int(s.get("frames_in_flight_at_window_end"))

        if total_rec is not None and completed is not None and overflow is not None and rejected is not None and cancelled is not None and in_flight is not None:
            sum_acc = completed + overflow + rejected + cancelled + in_flight
            if total_rec != sum_acc:
                errors.append(f"Rep {rep} Mech {mech}: Invariant mismatch. Received {total_rec} != completed ({completed}) + overflow ({overflow}) + rejected ({rejected}) + cancelled ({cancelled}) + in-flight ({in_flight}) [sum={sum_acc}]")

        # Causal timestamps ordering of milestones
        t_req = parse_int(s.get("request_timestamp_ns"))
        t_prep_start = parse_int(s.get("candidate_prep_start_ns"))
        t_prep_end = parse_int(s.get("candidate_prep_end_ns"))
        t_pub = parse_int(s.get("publication_timestamp_ns"))
        t_first_cand = parse_int(s.get("first_candidate_output_ns"))

        if t_req is not None and t_prep_start is not None:
            if t_req > t_prep_start:
                errors.append(f"Rep {rep} Mech {mech}: Request timestamp ({t_req}) > candidate prep start ({t_prep_start})")
        if t_prep_start is not None and t_prep_end is not None:
            if t_prep_start > t_prep_end:
                errors.append(f"Rep {rep} Mech {mech}: Candidate prep start ({t_prep_start}) > candidate prep end ({t_prep_end})")
        if t_prep_end is not None and t_pub is not None:
            if t_prep_end > t_pub:
                errors.append(f"Rep {rep} Mech {mech}: Candidate prep end ({t_prep_end}) > publication timestamp ({t_pub})")
        if t_pub is not None and t_first_cand is not None:
            if t_pub > t_first_cand:
                errors.append(f"Rep {rep} Mech {mech}: Publication timestamp ({t_pub}) > first candidate output timestamp ({t_first_cand})")

        # Reconfiguration intervals arithmetic
        req_to_eff = parse_int(s.get("request_to_effect_ns"))
        
        if req_to_eff is not None and t_first_cand is not None and t_req is not None:
            if req_to_eff != t_first_cand - t_req:
                errors.append(f"Rep {rep} Mech {mech}: request_to_effect_ns ({req_to_eff}) != first_candidate_output_ns ({t_first_cand}) - request_timestamp_ns ({t_req})")

        # Old plan frames completed during prep
        old_prep = parse_int(s.get("old_plan_frames_completed_during_prep"))
        if old_prep is not None and t_prep_start is not None and t_prep_end is not None:
            # Count actual old-plan frames completed during prep
            actual_old_prep = 0
            for fr in run_frames:
                p_ver = parse_int(fr.get("plan_version"))
                inside = parse_int(fr.get("inside_measurement_window")) == 1 or fr.get("inside_measurement_window") == "True"
                t_comp = parse_int(fr.get("completion_timestamp_ns"))
                if p_ver == 1 and inside and t_comp is not None:
                    if t_prep_start <= t_comp <= t_prep_end:
                        actual_old_prep += 1
            if old_prep != actual_old_prep:
                errors.append(f"Rep {rep} Mech {mech}: old_plan_frames_completed_during_prep ({old_prep}) != actual count ({actual_old_prep})")

        # Detector lifecycle after retirement
        act_det_ret = s.get("active_detector_id_after_retirement")
        cand_det = s.get("candidate_model_id")
        act_ver_ret = parse_int(s.get("active_plan_version_after_retirement"))
        act_det_count = parse_int(s.get("active_detector_instance_count_after_retirement"))
        live_plan_count = parse_int(s.get("live_plan_count_after_retirement"))
        live_det_count = parse_int(s.get("live_detector_count_after_retirement"))

        if act_det_ret and cand_det and act_det_ret != cand_det:
            errors.append(f"Rep {rep} Mech {mech}: active detector after retirement ({act_det_ret}) != candidate ({cand_det})")
        if act_ver_ret is not None and act_ver_ret != 2:
            errors.append(f"Rep {rep} Mech {mech}: active plan version after retirement ({act_ver_ret}) != 2")
        if act_det_count is not None and act_det_count != 1:
            errors.append(f"Rep {rep} Mech {mech}: active detector instance count after retirement ({act_det_count}) != 1")
        if live_plan_count is not None and live_plan_count != 1:
            errors.append(f"Rep {rep} Mech {mech}: live plan count after retirement ({live_plan_count}) != 1")
        if live_det_count is not None and live_det_count != 1:
            errors.append(f"Rep {rep} Mech {mech}: live detector count after retirement ({live_det_count}) != 1")

        # GPU memory stats consistency
        for prefix in ("gpu_memory_before_prep", "gpu_memory_during_coexistence", "gpu_memory_after_pub", "gpu_memory_after_ret"):
            alloc = parse_int(s.get(f"{prefix}_allocated_bytes"))
            resv = parse_int(s.get(f"{prefix}_reserved_bytes"))
            pk_alloc = parse_int(s.get(f"{prefix}_peak_allocated_bytes"))
            pk_resv = parse_int(s.get(f"{prefix}_peak_reserved_bytes"))

            if alloc is not None and alloc < 0:
                errors.append(f"Rep {rep} Mech {mech}: Negative allocated memory for {prefix}")
            if resv is not None and resv < 0:
                errors.append(f"Rep {rep} Mech {mech}: Negative reserved memory for {prefix}")
            if alloc is not None and resv is not None and alloc > resv:
                errors.append(f"Rep {rep} Mech {mech}: Allocated ({alloc}) > Reserved ({resv}) for {prefix}")
            if alloc is not None and pk_alloc is not None and alloc > pk_alloc:
                errors.append(f"Rep {rep} Mech {mech}: Allocated ({alloc}) > Peak Allocated ({pk_alloc}) for {prefix}")
            if resv is not None and pk_resv is not None and resv > pk_resv:
                errors.append(f"Rep {rep} Mech {mech}: Reserved ({resv}) > Peak Reserved ({pk_resv}) for {prefix}")

    # 3. Cross-mechanism validation within each repetition
    # Group summary rows by repetition
    summary_by_rep: dict[int, list[dict[str, str]]] = {}
    for s in summary_rows:
        rep = parse_int(s.get("repetition"))
        if rep is not None:
            summary_by_rep.setdefault(rep, []).append(s)

    for rep, s_list in summary_by_rep.items():
        if len(s_list) < 2:
            continue
        first = s_list[0]
        # Cross-mechanism matching fields
        match_fields = (
            "source_file_hash",
            "video_fps",
            "resolution",
            "measurement_start_media_frame_index",
            "measurement_start_media_pts_ns",
            "request_trigger_media_frame_index",
            "request_trigger_media_pts_ns",
            "pre_request_source_frame_target",
            "post_effect_source_frame_target",
        )

        for s in s_list[1:]:
            mech1, mech2 = first.get("mechanism"), s.get("mechanism")
            for field in match_fields:
                val1 = first.get(field)
                val2 = s.get(field)
                if val1 != val2:
                    errors.append(f"Rep {rep}: Cross-mechanism mismatch on field {field} between {mech1} ({val1}) and {mech2} ({val2})")

    # Output concise summary
    if not errors:
        print("PASS: Real-world benchmark result validation successful!")
        return True, errors
    else:
        print("FAIL: Real-world benchmark validation failed with the following errors:")
        for err in errors:
            print(f"  - {err}")
        return False, errors
