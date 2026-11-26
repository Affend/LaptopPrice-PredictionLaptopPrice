"""tabel awal: training_runs, model_versions, evaluation_results, predictions

Revision ID: f92a8397e1ef
Revises:
Create Date: 2026-11-26 10:02:14.706677
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "f92a8397e1ef"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# JSON biasa di SQLite, JSONB (biner, lebih cepat) di PostgreSQL
TIPE_JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    # 1 baris = 1 kali menjalankan `python -m ml.train` (id = nomor versi)
    op.create_table(
        "training_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dibuat_pada", sa.DateTime(timezone=True), nullable=False),
        sa.Column("durasi_detik", sa.Float(), nullable=False),
        sa.Column("ringkasan", TIPE_JSON, nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # 1 baris = 1 model terlatih dalam satu run
    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("training_run_id", sa.Integer(), nullable=False),
        sa.Column("nama_model", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=128), nullable=False),
        sa.Column("hyperparameter", TIPE_JSON, nullable=False),
        sa.Column("path_file", sa.String(length=512), nullable=False),
        sa.Column("dibuat_pada", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["training_run_id"], ["training_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("training_run_id", "nama_model", name="uq_model_per_run"),
    )
    with op.batch_alter_table("model_versions") as batch_op:
        batch_op.create_index("ix_model_versions_is_active", ["is_active"], unique=False)
        batch_op.create_index("ix_model_versions_training_run_id", ["training_run_id"], unique=False)

    # 1 baris = 1 jenis hasil evaluasi (metrik, cv, residual, ...) dari 1 model
    op.create_table(
        "evaluation_results",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=False),
        sa.Column("jenis", sa.String(length=32), nullable=False),
        sa.Column("data", TIPE_JSON, nullable=False),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_version_id", "jenis", name="uq_jenis_per_model"),
    )
    with op.batch_alter_table("evaluation_results") as batch_op:
        batch_op.create_index("ix_evaluation_results_jenis", ["jenis"], unique=False)
        batch_op.create_index("ix_evaluation_results_model_version_id", ["model_version_id"], unique=False)

    # 1 baris = 1 prediksi dari website
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=False),
        sa.Column("input", TIPE_JSON, nullable=False),
        sa.Column("harga_prediksi_eur", sa.Float(), nullable=False),
        sa.Column("harga_prediksi_idr", sa.Float(), nullable=False),
        sa.Column("dibuat_pada", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("predictions") as batch_op:
        batch_op.create_index("ix_predictions_dibuat_pada", ["dibuat_pada"], unique=False)
        batch_op.create_index("ix_predictions_model_version_id", ["model_version_id"], unique=False)


def downgrade() -> None:
    # Urutan kebalikan dari upgrade (tabel anak dihapus dulu)
    op.drop_table("predictions")
    op.drop_table("evaluation_results")
    op.drop_table("model_versions")
    op.drop_table("training_runs")
