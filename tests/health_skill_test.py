import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "data" / "skills" / "health" / "scripts" / "health"
LOADER = importlib.machinery.SourceFileLoader("tera_health_skill", str(SCRIPT))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
HEALTH = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(HEALTH)

DAY = date(2026, 9, 23)

# Stands in for the garminconnect package, in process and in the detached login
# worker, which finds it through PYTHONPATH.
FAKE_GARMINCONNECT = textwrap.dedent(
    """
    import json
    import os
    from pathlib import Path


    class GarminConnectConnectionError(Exception):
        pass


    class GarminConnectNotFoundError(GarminConnectConnectionError):
        pass


    class GarminConnectTooManyRequestsError(Exception):
        pass


    class GarminConnectAuthenticationError(Exception):
        pass


    class _Client:
        def dump(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "garmin_tokens.json").write_text(json.dumps({"di_token": "token"}))


    class Garmin:
        def __init__(self, email=None, password=None, return_on_mfa=False):
            self.email = email
            self.password = password
            self.client = _Client()

        def login(self, tokenstore=None):
            if tokenstore:
                if not (Path(tokenstore) / "garmin_tokens.json").exists():
                    raise GarminConnectAuthenticationError("Username and password are required")
                return None, None
            if self.password != "right-password":
                raise GarminConnectAuthenticationError("401 Unauthorized")
            if os.environ.get("FAKE_GARMIN_MFA"):
                return "needs_mfa", {"pending": True}
            return None, None

        def resume_login(self, state, code):
            if code != "123456":
                raise GarminConnectAuthenticationError("invalid code")
            return None, None

        def get_full_name(self):
            return "Test Owner"
    """
)


def load_fake_garminconnect():
    namespace = {}
    exec(compile(FAKE_GARMINCONNECT, "garminconnect", "exec"), namespace)
    return type("garminconnect", (), namespace)


def heart_rate_packet(bpm, *rr_ms):
    packet = bytes([0x10, bpm])
    for interval in rr_ms:
        packet += round(interval * 1024 / 1000).to_bytes(2, "little")
    return packet


def timestamp_bytes(moment):
    return moment.year.to_bytes(2, "little") + bytes(
        [moment.month, moment.day, moment.hour, moment.minute, moment.second]
    )


class HealthTestCase(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name).resolve()
        self.secrets = self.root / "secrets.json"
        self.environment = mock.patch.dict(
            os.environ,
            {"TERA_HEALTH_ROOT": str(self.root / "health"), "TERA_SECRETS_FILE": str(self.secrets)},
        )
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.tempdir.cleanup()

    def write_secrets(self, **values):
        secrets = {name: {"value": value, "set_at_ms": 0} for name, value in values.items()}
        self.secrets.write_text(json.dumps({"secrets": secrets}))

    def connection(self):
        return HEALTH.connect()

    def series(self, connection, name, values, end=DAY, source="garmin"):
        """Record one value a day, the last one on `end`."""
        for offset, value in enumerate(reversed(values)):
            HEALTH.record(connection, name, (end - timedelta(days=offset)).isoformat(), value, source)

    def topics(self, connection, day=DAY):
        return {(finding.level, finding.topic) for finding in HEALTH.recommendations(connection, day)}


class ExtractorTest(unittest.TestCase):
    def test_summary_weights_vigorous_minutes_and_drops_placeholders(self):
        triples = dict(
            (name, value)
            for name, _day, value in HEALTH.extract_summary(
                {
                    "totalSteps": 8500,
                    "moderateIntensityMinutes": 22,
                    "vigorousIntensityMinutes": 4,
                    "restingHeartRate": 54,
                    "averageStressLevel": -1,
                    "bodyBatteryHighestValue": 92,
                    "bodyBatteryChargedValue": 58,
                },
                "2026-09-23",
            )
        )
        self.assertEqual(triples["steps"], 8500)
        self.assertEqual(triples["intensity_minutes"], 30)
        self.assertIsNone(triples["stress_avg"])
        self.assertEqual(triples["body_battery_charged"], 58)

    def test_sleep_converts_to_hours_and_skips_empty_nights(self):
        payload = {
            "dailySleepDTO": {
                "sleepTimeSeconds": 25200,
                "deepSleepSeconds": 5400,
                "remSleepSeconds": 4800,
                "awakeSleepSeconds": 600,
                "avgSpO2": 96.0,
                "averageRespirationValue": 14.2,
                "sleepScores": {"overall": {"value": 84}},
            }
        }
        triples = {name: value for name, _day, value in HEALTH.extract_sleep(payload, "2026-09-23")}
        self.assertEqual(triples["sleep_hours"], 7)
        self.assertEqual(triples["sleep_deep_hours"], 1.5)
        self.assertEqual(triples["sleep_score"], 84)
        self.assertEqual(triples["spo2_sleep"], 96.0)
        self.assertEqual(HEALTH.extract_sleep({"dailySleepDTO": {"sleepTimeSeconds": None}}, "2026-09-23"), [])

    def test_hrv_handles_the_empty_no_content_response(self):
        self.assertEqual(
            HEALTH.extract_hrv({"hrvSummary": {"lastNightAvg": 52.0, "weeklyAvg": 48.5}}, "d"),
            [("hrv_night", "d", 52.0), ("hrv_weekly", "d", 48.5)],
        )
        self.assertEqual(HEALTH.extract_hrv({}, "d"), [("hrv_night", "d", None), ("hrv_weekly", "d", None)])

    def test_readiness_prefers_the_morning_value(self):
        payload = [
            {"score": 55, "timestamp": "2026-09-23T15:00:00.0", "inputContext": "UPDATE_REALTIME_VARIABLES"},
            {"score": 72, "timestamp": "2026-09-23T06:12:00.0", "inputContext": "AFTER_WAKEUP_RESET"},
        ]
        self.assertEqual(HEALTH.extract_readiness(payload, "d"), [("training_readiness", "d", 72)])
        self.assertEqual(HEALTH.extract_readiness([], "d"), [])

    def test_ranged_extractors_date_each_entry(self):
        self.assertEqual(
            HEALTH.extract_max_metrics([
                {"generic": {"calendarDate": "2026-09-20", "vo2MaxPreciseValue": 51.3}},
                {"generic": None},
            ]),
            [("vo2max", "2026-09-20", 51.3)],
        )
        moment = datetime(2026, 9, 23, 7, 30)
        weights = HEALTH.extract_body_composition(
            {"dateWeightList": [{"date": moment.timestamp() * 1000, "weight": 72400.0, "bodyFat": 18.5}]}
        )
        self.assertIn(("weight_kg", "2026-09-23T07:30:00", 72.4), weights)
        self.assertIn(("body_fat_pct", "2026-09-23T07:30:00", 18.5), weights)
        pressure = HEALTH.extract_blood_pressure({
            "measurementSummaries": [{"measurements": [
                {"systolic": 121, "diastolic": 79, "pulse": 60, "measurementTimestampLocal": "2026-09-23T08:00:00.0"}
            ]}]
        })
        self.assertEqual(pressure[0], ("bp_systolic", "2026-09-23T08:00:00", 121))

    def test_activity_row_converts_units(self):
        row = HEALTH.activity_row({
            "activityId": 19876543210,
            "activityName": "Morning Run",
            "startTimeLocal": "2026-09-23 06:30:00",
            "activityType": {"typeKey": "running"},
            "duration": 2400.0,
            "distance": 6200.0,
            "averageHR": 148.0,
            "activityTrainingLoad": 98.5,
        })
        self.assertEqual(row["id"], "19876543210")
        self.assertEqual(row["day"], "2026-09-23")
        self.assertEqual(row["started_at"], "2026-09-23T06:30:00")
        self.assertEqual(row["duration_min"], 40)
        self.assertEqual(row["distance_km"], 6.2)
        self.assertEqual(row["type"], "running")


class BluetoothDecoderTest(unittest.TestCase):
    def test_ieee_11073_floats(self):
        self.assertEqual(HEALTH.sfloat(0x0079), 121)
        self.assertEqual(HEALTH.sfloat(0xF06B), 10.7)
        self.assertEqual(HEALTH.sfloat(0xFFFF), -0.1)
        self.assertIsNone(HEALTH.sfloat(0x07FF))
        self.assertEqual(HEALTH.float32(0xFF00016E), 36.6)
        self.assertIsNone(HEALTH.float32(0x007FFFFF))

    def test_heart_rate_with_rr_intervals_and_16_bit_value(self):
        values, at, rr = HEALTH.decode_heart_rate(heart_rate_packet(72, 1000, 500))
        self.assertEqual(values, {"heart_rate": 72})
        self.assertIsNone(at)
        self.assertEqual(rr, [1000, 500])
        self.assertEqual(HEALTH.decode_heart_rate(bytes([0x01, 0x2C, 0x01]))[0], {"heart_rate": 300})

    def test_weight_in_kilograms_and_pounds(self):
        self.assertEqual(HEALTH.decode_weight(bytes([0x00, 0x90, 0x38]))[0], {"weight_kg": 72.4})
        moment = datetime(2026, 9, 23, 7, 30)
        values, at, _rr = HEALTH.decode_weight(bytes([0x03]) + (16000).to_bytes(2, "little") + timestamp_bytes(moment))
        self.assertAlmostEqual(values["weight_kg"], 72.575, places=3)
        self.assertEqual(at, moment)
        self.assertEqual(HEALTH.decode_weight(bytes([0x00, 0xFF, 0xFF]))[0], {})

    def test_blood_pressure_with_timestamp_pulse_and_kpa(self):
        moment = datetime(2026, 9, 23, 8, 0)
        packet = bytes([0x06, 0x79, 0x00, 0x51, 0x00, 0x5E, 0x00]) + timestamp_bytes(moment) + bytes([0x40, 0x00])
        values, at, _rr = HEALTH.decode_blood_pressure(packet)
        self.assertEqual(values, {"bp_systolic": 121, "bp_diastolic": 81, "pulse": 64})
        self.assertEqual(at, moment)
        values, _at, _rr = HEALTH.decode_blood_pressure(bytes([0x01, 0xA0, 0xF0, 0x6B, 0xF0, 0x00, 0x00]))
        self.assertEqual(values, {"bp_systolic": 120.0, "bp_diastolic": 80.3})

    def test_temperature_in_celsius_and_fahrenheit(self):
        self.assertEqual(HEALTH.decode_temperature(bytes([0x00, 0x6E, 0x01, 0x00, 0xFF]))[0], {"body_temp_c": 36.6})
        self.assertEqual(HEALTH.decode_temperature(bytes([0x01, 0xDA, 0x03, 0x00, 0xFF]))[0], {"body_temp_c": 37.0})

    def test_pulse_oximeter(self):
        self.assertEqual(HEALTH.decode_spot_check(bytes([0x00, 0x61, 0x00, 0x46, 0x00]))[0], {"spo2": 97, "pulse": 70})
        self.assertEqual(HEALTH.decode_continuous(bytes([0x00, 0x62, 0x00, 0xFF, 0x07]))[0], {"spo2": 98})

    def test_truncated_payload_is_an_error_not_a_crash(self):
        with self.assertRaisesRegex(HEALTH.HealthError, "truncated"):
            HEALTH.decode_blood_pressure(bytes([0x00, 0x79]))

    def test_stream_summary_drops_artefacts_before_rmssd(self):
        rr = [800, 850] * 15 + [3000]
        summary = HEALTH.summarise_stream([{"heart_rate": 70}, {"heart_rate": 74}], rr)
        self.assertEqual(summary, {"heart_rate": 72, "hrv_rmssd": 50})
        self.assertNotIn("hrv_rmssd", HEALTH.summarise_stream([{"heart_rate": 70}], [800, 850]))

    def test_kinds_come_from_advertised_services(self):
        services = ["0000180D-0000-1000-8000-00805F9B34FB", "0000180f-0000-1000-8000-00805f9b34fb"]
        self.assertEqual(HEALTH.kinds_for(services), ["hr"])
        self.assertEqual(HEALTH.kinds_for([HEALTH.ble_uuid(0x1822)]), ["spo2", "spo2-live"])


class StorageTest(HealthTestCase):
    def test_readings_replace_in_place_and_average_per_day(self):
        connection = self.connection()
        HEALTH.record(connection, "weight_kg", datetime(2026, 9, 23, 7), 72.0, "manual")
        HEALTH.record(connection, "weight_kg", datetime(2026, 9, 23, 7), 72.4, "manual")
        HEALTH.record(connection, "weight_kg", datetime(2026, 9, 23, 21), 73.0, "ble")
        HEALTH.record(connection, "weight_kg", datetime(2026, 9, 23, 22), None, "ble")
        count = connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        self.assertEqual(count, 2)
        self.assertAlmostEqual(HEALTH.daily_series(connection, "weight_kg", DAY, DAY)[DAY], 72.7)

    def test_unknown_metric_is_rejected(self):
        with self.assertRaisesRegex(HEALTH.HealthError, "Unknown metric"):
            HEALTH.record(self.connection(), "mood", "2026-09-23", 5, "manual")

    def test_database_is_private(self):
        self.connection()
        self.assertEqual((self.root / "health").stat().st_mode & 0o777, 0o700)

    def test_sync_range_backfills_then_catches_up(self):
        connection = self.connection()
        self.assertEqual(HEALTH.sync_range(connection, None, DAY), (DAY - timedelta(days=29), DAY))
        self.assertEqual(HEALTH.sync_range(connection, 3, DAY), (DAY - timedelta(days=2), DAY))
        HEALTH.set_meta(connection, "garmin_last_day", "2026-09-20")
        self.assertEqual(HEALTH.sync_range(connection, None, DAY), (date(2026, 9, 19), DAY))
        HEALTH.set_meta(connection, "garmin_last_day", "2026-09-23")
        self.assertEqual(HEALTH.sync_range(connection, None, DAY), (date(2026, 9, 22), DAY))

    def test_parse_entries(self):
        self.assertEqual(HEALTH.parse_entries(["weight_kg=72.4", "pulse=60"]), [("weight_kg", 72.4), ("pulse", 60.0)])
        for bad in ("weight_kg", "weight_kg=heavy", "mood=5"):
            with self.subTest(bad=bad), self.assertRaises(HEALTH.HealthError):
                HEALTH.parse_entries([bad])


class FakeGarminClient:
    """Answers like the Garmin API for two days, the older one with no summary yet."""

    def __init__(self, garminconnect):
        self.garminconnect = garminconnect
        self.calls = []

    def __getattr__(self, method):
        def call(*args):
            self.calls.append((method, args))
            return self.answer(method, *args)

        return call

    def answer(self, method, *args):
        if method == "get_user_summary":
            if args[0] == "2026-09-22":
                raise self.garminconnect.GarminConnectConnectionError("No data received from server")
            return {"totalSteps": 9000, "restingHeartRate": 55}
        if method == "get_sleep_data":
            return {"dailySleepDTO": {"sleepTimeSeconds": 27000}}
        if method == "get_hrv_data":
            return {}
        if method == "get_training_readiness":
            raise self.garminconnect.GarminConnectNotFoundError("404")
        if method == "get_max_metrics_range":
            return [{"generic": {"calendarDate": "2026-09-22", "vo2MaxPreciseValue": 51.3}}]
        if method in ("get_body_composition", "get_blood_pressure"):
            return {}
        if method == "get_activities_by_date":
            return [{"activityId": 1, "startTimeLocal": "2026-09-23 06:30:00", "duration": 1800.0}]
        raise AssertionError(f"unexpected call {method}")


class SyncTest(HealthTestCase):
    def setUp(self):
        super().setUp()
        self.garminconnect = load_fake_garminconnect()
        self.client = FakeGarminClient(self.garminconnect)
        for target, value in (
            ("import_garmin", lambda: self.garminconnect),
            ("garmin_client", lambda: self.client),
            ("REQUEST_PAUSE_SECONDS", 0),
        ):
            patcher = mock.patch.object(HEALTH, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_sync_stores_values_skips_empty_days_and_records_progress(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            HEALTH.sync(2, DAY)
        self.assertIn("Synced 2026-09-22 to 2026-09-23", output.getvalue())
        connection = self.connection()
        self.assertEqual(HEALTH.value_on(connection, "steps", DAY), 9000)
        self.assertIsNone(HEALTH.value_on(connection, "steps", DAY - timedelta(days=1)))
        self.assertEqual(HEALTH.value_on(connection, "sleep_hours", DAY), 7.5)
        self.assertEqual(HEALTH.value_on(connection, "vo2max", date(2026, 9, 22)), 51.3)
        self.assertEqual(HEALTH.get_meta(connection, "garmin_last_day"), "2026-09-23")
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM activities").fetchone()[0], 1)
        stored = {row[0] for row in connection.execute("SELECT kind FROM raw")}
        self.assertEqual(stored, {"summary", "sleep", "max_metrics"})

    def test_rate_limit_stops_the_sync(self):
        def limited(method, *args):
            raise self.garminconnect.GarminConnectTooManyRequestsError("429")

        self.client.answer = limited
        with self.assertRaisesRegex(HEALTH.HealthError, "rate limiting"):
            HEALTH.sync(1, DAY)

    def test_write_rate_limit_is_recognised(self):
        def limited(method, *args):
            raise self.garminconnect.GarminConnectConnectionError("API Error 429: Too Many Requests")

        self.client.answer = limited
        with self.assertRaisesRegex(HEALTH.HealthError, "rate limiting"):
            HEALTH.call_garmin(self.client, "add_weigh_in", 72.4)

    def test_log_pushes_weight_and_blood_pressure_to_garmin(self):
        self.client.answer = lambda method, *args: None
        with contextlib.redirect_stdout(io.StringIO()) as output:
            HEALTH.log_entries(["weight_kg=72.4", "bp_systolic=121", "bp_diastolic=79", "pulse=60"],
                               "2026-09-23T07:30", to_garmin=True)
        self.assertEqual(self.client.calls, [
            ("add_weigh_in", (72.4, "kg", "2026-09-23T07:30:00")),
            ("set_blood_pressure", (121, 79, 60, "2026-09-23T07:30:00")),
        ])
        self.assertIn("Sent weight and blood pressure to Garmin Connect.", output.getvalue())

    def test_blood_pressure_without_pulse_leaves_it_out(self):
        self.client.answer = lambda method, *args: None
        with contextlib.redirect_stdout(io.StringIO()):
            HEALTH.log_entries(["bp_systolic=121", "bp_diastolic=79"], "2026-09-23T07:30", to_garmin=True)
        self.assertEqual(self.client.calls, [("set_blood_pressure", (121, 79, None, "2026-09-23T07:30:00"))])


class AnalysisTest(HealthTestCase):
    def test_baseline_needs_a_week_of_history(self):
        connection = self.connection()
        self.series(connection, "hrv_night", [50] * 6, end=DAY - timedelta(days=1))
        self.assertIsNone(HEALTH.baseline(connection, "hrv_night", DAY))
        self.series(connection, "hrv_night", [48, 52] * 14, end=DAY - timedelta(days=1))
        normal = HEALTH.baseline(connection, "hrv_night", DAY)
        self.assertEqual((normal.mean, normal.stdev, normal.count), (50, 2, 28))
        self.assertEqual(HEALTH.deviation(46, normal), -2)
        self.assertIsNone(HEALTH.deviation(46, HEALTH.Baseline(50, 0, 28)))

    def test_poor_recovery_warns_and_good_recovery_says_so(self):
        connection = self.connection()
        self.series(connection, "hrv_night", [48, 52] * 14 + [38])
        self.series(connection, "resting_hr", [54, 56] * 14 + [62])
        self.assertIn(("warn", "recovery"), self.topics(connection))

        self.series(connection, "hrv_night", [56])
        self.series(connection, "resting_hr", [55])
        self.series(connection, "training_readiness", [80])
        self.assertIn(("good", "recovery"), self.topics(connection))

    def test_blood_pressure_categories(self):
        connection = self.connection()
        for offset, (systolic, diastolic) in enumerate([(144, 92), (146, 91), (142, 93)]):
            at = datetime(2026, 9, 21 + offset, 8)
            HEALTH.record(connection, "bp_systolic", at, systolic, "manual")
            HEALTH.record(connection, "bp_diastolic", at, diastolic, "manual")
        [finding] = HEALTH.blood_pressure_findings(connection, DAY)
        self.assertEqual(finding.level, "warn")
        self.assertIn("stage 2", finding.text)

        HEALTH.record(connection, "bp_systolic", datetime(2026, 9, 23, 20), 185, "ble")
        HEALTH.record(connection, "bp_diastolic", datetime(2026, 9, 23, 20), 100, "ble")
        [finding] = HEALTH.blood_pressure_findings(connection, DAY)
        self.assertEqual(finding.level, "alert")

    def test_short_sleep_low_activity_and_fever(self):
        connection = self.connection()
        self.series(connection, "sleep_hours", [6.5] * 6 + [5.2])
        self.series(connection, "intensity_minutes", [10] * 7)
        self.series(connection, "steps", [3000] * 7)
        HEALTH.record(connection, "body_temp_c", datetime(2026, 9, 23, 9), 38.4, "ble")
        sleep = [finding.text for finding in HEALTH.sleep_findings(connection, DAY)]
        self.assertEqual(len(sleep), 2)
        self.assertIn("5.2 h", sleep[0])
        self.assertEqual(
            {topic for _level, topic in self.topics(connection)},
            {"sleep", "activity", "temperature"},
        )

    def test_training_load_spike(self):
        connection = self.connection()
        for offset in range(28):
            day = DAY - timedelta(days=offset)
            load = 60 if offset < 7 else 15
            HEALTH.store_activity(connection, {"id": str(offset), "day": day.isoformat(),
                                               "started_at": day.isoformat(), "training_load": load})
        ratio, sessions = HEALTH.training_load_ratio(connection, DAY)
        self.assertEqual(sessions, 7)
        self.assertAlmostEqual(ratio, 420 / ((420 + 315) / 4))
        topics = self.topics(connection)
        self.assertIn(("warn", "training"), topics)
        self.assertIn(("info", "training"), topics)

    def test_findings_sort_most_urgent_first(self):
        connection = self.connection()
        self.series(connection, "body_battery_high", [40])
        HEALTH.record(connection, "bp_systolic", datetime(2026, 9, 23, 8), 190, "manual")
        HEALTH.record(connection, "bp_diastolic", datetime(2026, 9, 23, 8), 110, "manual")
        levels = [finding.level for finding in HEALTH.recommendations(connection, DAY)]
        self.assertEqual(levels, ["alert", "info"])

    def test_stale_watch_data_is_flagged(self):
        connection = self.connection()
        HEALTH.set_meta(connection, "garmin_synced_at", "2026-09-23T08:00:00")
        self.series(connection, "steps", [8000], end=DAY - timedelta(days=3))
        self.assertIn(("info", "data"), self.topics(connection))

    def test_trends_compare_the_last_week_with_the_month_before(self):
        connection = self.connection()
        self.series(connection, "resting_hr", [60] * 28 + [54] * 7)
        self.series(connection, "sleep_hours", [7.0] * 35)
        self.series(connection, "weight_kg", [70] * 28 + [74] * 7)
        self.series(connection, "steps", [9000] * 3)
        rows = {row["metric"]: row for row in HEALTH.trend_rows(connection, DAY)}
        self.assertEqual(rows["resting_hr"]["verdict"], "improving")
        self.assertEqual(rows["resting_hr"]["change_pct"], -10)
        self.assertEqual(rows["sleep_hours"]["verdict"], "steady")
        self.assertEqual(rows["weight_kg"]["verdict"], "up")
        self.assertEqual(rows["steps"]["verdict"], "new")
        self.assertEqual(list(rows), ["steps", "resting_hr", "sleep_hours", "weight_kg"])

    def test_brief_json_has_values_baselines_and_recommendations(self):
        connection = self.connection()
        self.series(connection, "hrv_night", [48, 52] * 14 + [38])
        self.series(connection, "sleep_hours", [7.5])
        data = HEALTH.brief_data(connection, DAY)
        self.assertEqual(data["metrics"]["hrv_night"], {"value": 38, "usual": 50, "deviation": -6})
        self.assertEqual(data["metrics"]["sleep_hours"]["usual"], None)
        self.assertEqual(data["recommendations"][0]["topic"], "recovery")
        json.dumps(data)


class CommandLineTest(HealthTestCase):
    """Runs the script as tera would, including the detached Garmin login worker."""

    def setUp(self):
        super().setUp()
        package = self.root / "fake" / "garminconnect"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(FAKE_GARMINCONNECT)
        self.env = dict(os.environ, PYTHONPATH=str(self.root / "fake"))

    def tearDown(self):
        # Never leave a worker behind waiting for a code.
        state = HEALTH.read_json(HEALTH.login_state_path())
        if state and state.get("pid") and HEALTH.process_alive(state["pid"]):
            os.kill(state["pid"], 15)
        super().tearDown()

    def health(self, *arguments, **extra_env):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *arguments],
            capture_output=True, text=True, timeout=60, env=dict(self.env, **extra_env),
        )

    def test_status_log_brief_and_trends_on_a_fresh_workspace(self):
        status = self.health("status")
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertIn("garmin-credentials: missing", status.stdout)
        self.assertIn("database: empty", status.stdout)

        logged = self.health("log", "weight_kg=72.4", "--at", "2026-09-23T07:30")
        self.assertEqual(logged.returncode, 0, logged.stderr)
        self.assertIn("Logged weight_kg 72.4 at 2026-09-23T07:30:00.", logged.stdout)

        brief = self.health("brief", "--date", "2026-09-23", "--json")
        self.assertEqual(json.loads(brief.stdout)["metrics"]["weight_kg"]["value"], 72.4)
        trends = self.health("trends", "--date", "2026-09-23")
        self.assertIn("weight: 72.4 kg, nothing earlier to compare", trends.stdout)
        history = self.health("history", "weight_kg", "--days", "1")
        self.assertEqual(history.returncode, 0, history.stderr)

        bad = self.health("log", "mood=5")
        self.assertEqual(bad.returncode, 1)
        self.assertTrue(bad.stderr.startswith("health: Unknown metric mood."))

    def test_login_asks_for_missing_secrets(self):
        result = self.health("login")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Missing GARMIN_EMAIL and GARMIN_PASSWORD", result.stderr)
        self.assertIn("request_secret", result.stderr)

    def test_login_without_mfa_saves_a_private_token(self):
        self.write_secrets(GARMIN_EMAIL="owner@example.com", GARMIN_PASSWORD="right-password")
        result = self.health("login")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Logged in to Garmin as Test Owner.", result.stdout)
        self.assertTrue((HEALTH.token_dir() / "garmin_tokens.json").exists())
        self.assertIn("garmin-login: complete", self.health("status").stdout)
        self.assertEqual(HEALTH.login_state_path().stat().st_mode & 0o777, 0o600)
        self.assertNotIn("right-password", HEALTH.login_state_path().read_text())

    def test_login_reports_a_wrong_password(self):
        self.write_secrets(GARMIN_EMAIL="owner@example.com", GARMIN_PASSWORD="wrong")
        result = self.health("login")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Garmin rejected the email or password", result.stderr)
        self.assertIn("garmin-login: failed", self.health("status").stdout)

    def test_mfa_code_arrives_in_a_later_process(self):
        self.write_secrets(GARMIN_EMAIL="owner@example.com", GARMIN_PASSWORD="right-password")
        started = self.health("login", FAKE_GARMIN_MFA="1")
        self.assertEqual(started.returncode, 0, started.stderr)
        self.assertIn("health login --mfa CODE", started.stdout)
        self.assertIn("garmin-login: waiting-for-code", self.health("status").stdout)

        again = self.health("login", FAKE_GARMIN_MFA="1")
        self.assertEqual(again.returncode, 1)
        self.assertIn("already waiting", again.stderr)

        wrong = self.health("login", "--mfa", "000000")
        self.assertEqual(wrong.returncode, 0, wrong.stderr)
        self.assertIn("Garmin rejected that code.", wrong.stdout)

        right = self.health("login", "--mfa", " 123456 ")
        self.assertEqual(right.returncode, 0, right.stderr)
        self.assertIn("Logged in to Garmin as Test Owner.", right.stdout)
        self.assertFalse(HEALTH.login_code_path().exists())

        stray = self.health("login", "--mfa", "123456")
        self.assertEqual(stray.returncode, 1)
        self.assertIn("No Garmin login is waiting", stray.stderr)

    def test_logout_stops_a_waiting_worker(self):
        self.write_secrets(GARMIN_EMAIL="owner@example.com", GARMIN_PASSWORD="right-password")
        self.assertEqual(self.health("login", FAKE_GARMIN_MFA="1").returncode, 0)
        pid = HEALTH.read_json(HEALTH.login_state_path())["pid"]
        self.assertIn("Logged out of Garmin.", self.health("logout").stdout)
        deadline = time.monotonic() + 10
        while HEALTH.process_alive(pid) and time.monotonic() < deadline:
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
            time.sleep(0.1)
        self.assertFalse(HEALTH.process_alive(pid))
        self.assertIn("garmin-login: missing", self.health("status").stdout)


if __name__ == "__main__":
    unittest.main()
