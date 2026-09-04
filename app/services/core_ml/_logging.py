"""
Konsolidasi logging core_ml — pakai standard library logging.
(Menggantikan core_ml/logger.py riset yang dipindah ke research/)
"""
import logging

logger = logging.getLogger("app.services.core_ml")
