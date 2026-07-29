from dataclasses import dataclass

RECORD_URL = "https://repository.cern/records/vy9x7-bzn92/files"
"""Base URL of the CERN Open Data record."""


@dataclass(frozen=True)
class Archive:
    """One downloadable archive in the data release."""

    name: str
    md5: str
    size: int | None = None


ARCHIVES = {
    "K-Pi.Kpi_Kpi.P0.tar.gz": Archive(
        "K-Pi.Kpi_Kpi.P0.tar.gz", "d0f7ca840bc20fc89248198fa4178181", 43815300160
    ),
    "K-Pi.Kpi_Kpi.P1.tar.gz": Archive(
        "K-Pi.Kpi_Kpi.P1.tar.gz", "3ca65f2f35bb1b5cf92ab20b0d8295d1", 96596759824
    ),
    "K-Pi.Kpi_Kpi.P2.tar.gz": Archive(
        "K-Pi.Kpi_Kpi.P2.tar.gz", "04a7260116414acbc22b8107b77a7380", 123650117697
    ),
    "K-Pi.Kpi_Kpi.P3.tar.gz": Archive(
        "K-Pi.Kpi_Kpi.P3.tar.gz", "13e5ab681f41608e855650ecba2f317d", 63108563707
    ),
    "K-Pi.Kpi_Kpi.P4.tar.gz": Archive(
        "K-Pi.Kpi_Kpi.P4.tar.gz", "91813883add0a7acc1025ea4a8c1c8bf", 29221352871
    ),
    "K-Pi.Kpi_V.tar.gz": Archive(
        "K-Pi.Kpi_V.tar.gz", "22a80de37b92f0ee2ee416df3130fca6", 21141395823
    ),
    "K-Pi.V_V.tar.gz": Archive(
        "K-Pi.V_V.tar.gz", "191d50a2c447ffc60d4fbc9ce96391fa", 3921227054
    ),
    "Kaon.tar.gz": Archive(
        "Kaon.tar.gz", "cd33160f0b684c5152a387f757cb5ea6", 447746910
    ),
    "Pi-Pi.Pipi_Pipi.P0.tar.gz": Archive(
        "Pi-Pi.Pipi_Pipi.P0.tar.gz", "3ae7d74d8c2edf5de7ebd6c968fee7af", 42715339007
    ),
    "Pi-Pi.Pipi_Pipi.P1.tar.gz": Archive(
        "Pi-Pi.Pipi_Pipi.P1.tar.gz", "118586ac04c7b64dc510585e59e9015f", 95604812401
    ),
    "Pi-Pi.Pipi_Pipi.P2.tar.gz": Archive(
        "Pi-Pi.Pipi_Pipi.P2.tar.gz", "e66ad7d2ce135955760b567b3712204a", 122372750220
    ),
    "Pi-Pi.Pipi_Pipi.P3.tar.gz": Archive(
        "Pi-Pi.Pipi_Pipi.P3.tar.gz", "90e1a48c6c7dd1fe629e6b52c0fdf97e", 62463540024
    ),
    "Pi-Pi.Pipi_Pipi.P4.tar.gz": Archive(
        "Pi-Pi.Pipi_Pipi.P4.tar.gz", "950e038823ca21b98b616a96d4a007bd", 28911975945
    ),
    "Pi-Pi.Pipi_V.tar.gz": Archive(
        "Pi-Pi.Pipi_V.tar.gz", "633ef7cd1fe05ee61f06be0944808d76", 20860667502
    ),
    "Pi-Pi.V_V.tar.gz": Archive(
        "Pi-Pi.V_V.tar.gz", "0dbd91a137219f0b90e539265216d5f1", 3869581655
    ),
    "Pion.tar.gz": Archive(
        "Pion.tar.gz", "c3278b304dd65a18092bba145a7af654", 444195861
    ),
}

IRREP_ARCHIVES = {
    "K-Pi": {
        "T1u[000]": (
            "K-Pi.Kpi_Kpi.P0.tar.gz",
            "K-Pi.Kpi_V.tar.gz",
            "K-Pi.V_V.tar.gz",
        ),
        "E[001]": (
            "K-Pi.Kpi_Kpi.P1.tar.gz",
            "K-Pi.Kpi_V.tar.gz",
            "K-Pi.V_V.tar.gz",
        ),
        "B1[110]": (
            "K-Pi.Kpi_Kpi.P2.tar.gz",
            "K-Pi.Kpi_V.tar.gz",
            "K-Pi.V_V.tar.gz",
        ),
        "B2[110]": (
            "K-Pi.Kpi_Kpi.P2.tar.gz",
            "K-Pi.Kpi_V.tar.gz",
            "K-Pi.V_V.tar.gz",
        ),
        "E[111]": (
            "K-Pi.Kpi_Kpi.P3.tar.gz",
            "K-Pi.Kpi_V.tar.gz",
            "K-Pi.V_V.tar.gz",
        ),
        "E[002]": (
            "K-Pi.Kpi_Kpi.P4.tar.gz",
            "K-Pi.Kpi_V.tar.gz",
            "K-Pi.V_V.tar.gz",
        ),
    },
    "Pi-Pi": {
        "T1u[000]": (
            "Pi-Pi.Pipi_Pipi.P0.tar.gz",
            "Pi-Pi.Pipi_V.tar.gz",
            "Pi-Pi.V_V.tar.gz",
        ),
        "E[001]": (
            "Pi-Pi.Pipi_Pipi.P1.tar.gz",
            "Pi-Pi.Pipi_V.tar.gz",
            "Pi-Pi.V_V.tar.gz",
        ),
        "B1[110]": (
            "Pi-Pi.Pipi_Pipi.P2.tar.gz",
            "Pi-Pi.Pipi_V.tar.gz",
            "Pi-Pi.V_V.tar.gz",
        ),
        "B2[110]": (
            "Pi-Pi.Pipi_Pipi.P2.tar.gz",
            "Pi-Pi.Pipi_V.tar.gz",
            "Pi-Pi.V_V.tar.gz",
        ),
        "E[111]": (
            "Pi-Pi.Pipi_Pipi.P3.tar.gz",
            "Pi-Pi.Pipi_V.tar.gz",
            "Pi-Pi.V_V.tar.gz",
        ),
        "E[002]": (
            "Pi-Pi.Pipi_Pipi.P4.tar.gz",
            "Pi-Pi.Pipi_V.tar.gz",
            "Pi-Pi.V_V.tar.gz",
        ),
        "A1[001]": (
            "Pi-Pi.Pipi_Pipi.P1.tar.gz",
            "Pi-Pi.Pipi_V.tar.gz",
            "Pi-Pi.V_V.tar.gz",
        ),
        "A1[110]": (
            "Pi-Pi.Pipi_Pipi.P2.tar.gz",
            "Pi-Pi.Pipi_V.tar.gz",
            "Pi-Pi.V_V.tar.gz",
        ),
        "A1[111]": (
            "Pi-Pi.Pipi_Pipi.P3.tar.gz",
            "Pi-Pi.Pipi_V.tar.gz",
            "Pi-Pi.V_V.tar.gz",
        ),
        "A1[002]": (
            "Pi-Pi.Pipi_Pipi.P4.tar.gz",
            "Pi-Pi.Pipi_V.tar.gz",
            "Pi-Pi.V_V.tar.gz",
        ),
    },
}
