import logging

from mockingbird.ui.log_bridge import QtLogHandler


def test_handler_forwards_formatted_record():
    lines = []
    handler = QtLogHandler(lines.append)
    logger = logging.getLogger("test.mockingbird.bridge")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        logger.info("hello %s", "world")
        logger.debug("hidden at info level")
    finally:
        logger.removeHandler(handler)
    assert len(lines) == 1
    assert "hello world" in lines[0]
