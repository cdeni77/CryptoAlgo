"""Create canonical trades/signals/wallet schema.

Revision ID: 20260213_000001
Revises:
Create Date: 2026-02-13 00:00:01
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20260213_000001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


tradeside_enum = sa.Enum("LONG", "SHORT", name="tradeside")
tradestatus_enum = sa.Enum("OPEN", "CLOSED", name="tradestatus")


def upgrade() -> None:
    bind = op.get_bind()
    tradeside_enum.create(bind, checkfirst=True)
    tradestatus_enum.create(bind, checkfirst=True)

    op.create_table(
        "trades",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("coin", sa.String(), nullable=False),
        sa.Column("datetime_open", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("datetime_close", sa.DateTime(timezone=True), nullable=True),
        sa.Column("side", tradeside_enum, nullable=False),
        sa.Column("contracts", sa.Float(), nullable=False),
        sa.Column("entry_price", sa.Float(), nullable=False),
        sa.Column("exit_price", sa.Float(), nullable=True),
        sa.Column("fee_open", sa.Float(), nullable=True),
        sa.Column("fee_close", sa.Float(), nullable=True),
        sa.Column("net_pnl", sa.Float(), nullable=True),
        sa.Column("margin_used", sa.Float(), nullable=True),
        sa.Column("leverage", sa.Float(), nullable=True),
        sa.Column("reason_entry", sa.Text(), nullable=True),
        sa.Column("reason_exit", sa.Text(), nullable=True),
        sa.Column("status", tradestatus_enum, nullable=False),
    )
    op.create_index("ix_trades_id", "trades", ["id"], unique=False)
    op.create_index("ix_trades_coin", "trades", ["coin"], unique=False)

    op.create_table(
        "signals",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("coin", sa.String(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("direction", sa.String(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("raw_probability", sa.Float(), nullable=True),
        sa.Column("model_auc", sa.Float(), nullable=True),
        sa.Column("price_at_signal", sa.Float(), nullable=True),
        sa.Column("momentum_pass", sa.Boolean(), nullable=True),
        sa.Column("trend_pass", sa.Boolean(), nullable=True),
        sa.Column("regime_pass", sa.Boolean(), nullable=True),
        sa.Column("ml_pass", sa.Boolean(), nullable=True),
        sa.Column("contracts_suggested", sa.Integer(), nullable=True),
        sa.Column("notional_usd", sa.Float(), nullable=True),
        sa.Column("acted_on", sa.Boolean(), nullable=True),
        sa.Column("trade_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
    )
    op.create_index("ix_signals_id", "signals", ["id"], unique=False)
    op.create_index("ix_signals_coin", "signals", ["coin"], unique=False)
    op.create_index("ix_signals_timestamp", "signals", ["timestamp"], unique=False)

    op.create_table(
        "wallet",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("balance", sa.Float(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("wallet")

    op.drop_index("ix_signals_timestamp", table_name="signals")
    op.drop_index("ix_signals_coin", table_name="signals")
    op.drop_index("ix_signals_id", table_name="signals")
    op.drop_table("signals")

    op.drop_index("ix_trades_coin", table_name="trades")
    op.drop_index("ix_trades_id", table_name="trades")
    op.drop_table("trades")

    bind = op.get_bind()
    tradestatus_enum.drop(bind, checkfirst=True)
    tradeside_enum.drop(bind, checkfirst=True)
