import sys
from pathlib import Path

from PySide6.QtCore import (
    QObject,
    QThread,
    Signal,
    Slot,
    Property,
    QUrl,
)

from PySide6.QtGui import (
    QGuiApplication,
    QIcon,
    QDesktopServices,
)

from PySide6.QtQml import (
    QQmlApplicationEngine,
)

from PySide6.QtQuickControls2 import (
    QQuickStyle,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR)
)

QML_FILE = (
    PROJECT_ROOT
    / "ui"
    / "qml"
    / "Main.qml"
)

ICON_FILE = (
    PROJECT_ROOT
    / "ui"
    / "assets"
    / "rag_icon.svg"
)


# ============================================================
# Worker
# ============================================================

from rag_worker import (
    RAGWorker,
)


# ============================================================
# UI Controller
# ============================================================

class RAGController(QObject):

    # ========================================================
    # Worker -> UI signals
    # ========================================================

    ready = Signal()

    statusChanged = Signal(str)

    errorOccurred = Signal(str)

    answerStarted = Signal()

    answerToken = Signal(str)

    answerFinished = Signal()

    sourcesChanged = Signal(str)

    imagesChanged = Signal(str)

    tablesChanged = Signal(str)

    documentsChanged = Signal(str)

    # --------------------------------------------------------
    # NEW:
    # Tell QML to clear the current chat.
    # --------------------------------------------------------

    chatCleared = Signal()


    # ========================================================
    # UI -> Worker signals
    # ========================================================

    initializeWorker = Signal()

    askWorker = Signal(
        str,
        str,
    )

    addDocumentWorker = Signal(
        str,
    )

    removeDocumentWorker = Signal(
        str,
    )

    shutdownWorker = Signal()


    # ========================================================
    # Properties
    # ========================================================

    @Property(
        str,
        constant=True,
    )
    def documentsPath(self):

        return (
            PROJECT_ROOT
            / "documents"
        ).as_uri()


    @Property(
        str,
        constant=True,
    )
    def defaultRagPrompt(self):

        return (
            "Answer the user's question using "
            "only the provided context.\n"
            "If the answer cannot be found in "
            "the context, say that there is not "
            "enough information in the provided "
            "documents.\n"
            "Do not invent facts."
        )


    # ========================================================
    # Constructor
    # ========================================================

    def __init__(
        self,
        worker,
        parent=None,
    ):

        super().__init__(parent)

        self.worker = worker


        # ----------------------------------------------------
        # Worker -> UI
        # ----------------------------------------------------

        worker.ready.connect(
            self.ready
        )

        worker.statusChanged.connect(
            self.statusChanged
        )

        worker.errorOccurred.connect(
            self.errorOccurred
        )

        worker.answerStarted.connect(
            self.answerStarted
        )

        worker.answerToken.connect(
            self.answerToken
        )

        worker.answerFinished.connect(
            self.answerFinished
        )

        worker.sourcesChanged.connect(
            self.sourcesChanged
        )

        worker.imagesChanged.connect(
            self.imagesChanged
        )

        worker.tablesChanged.connect(
            self.tablesChanged
        )

        worker.documentsChanged.connect(
            self.documentsChanged
        )


        # ----------------------------------------------------
        # UI -> Worker
        # ----------------------------------------------------

        self.initializeWorker.connect(
            worker.initialize
        )

        self.askWorker.connect(
            worker.ask
        )

        self.addDocumentWorker.connect(
            worker.add_document
        )

        self.removeDocumentWorker.connect(
            worker.remove_document
        )

        self.shutdownWorker.connect(
            worker.shutdown
        )


    # ========================================================
    # Initialize
    # ========================================================

    @Slot()
    def initialize(self):

        self.initializeWorker.emit()


    # ========================================================
    # Ask
    # ========================================================

    @Slot(str, str)
    def ask(
        self,
        query,
        rag_prompt,
    ):

        self.askWorker.emit(
            query,
            rag_prompt,
        )


    # ========================================================
    # Add document
    # ========================================================

    @Slot(str)
    def addDocument(
        self,
        file_path,
    ):

        if not file_path:
            return


        # QML FileDialog returns a URL.
        if file_path.startswith(
            "file:///"
        ):

            url = QUrl(
                file_path
            )

            local_path = (
                url.toLocalFile()
            )

        else:

            local_path = file_path


        self.addDocumentWorker.emit(
            local_path
        )


    # ========================================================
    # Remove document
    # ========================================================

    @Slot(str)
    def removeDocument(
        self,
        document_name,
    ):

        if not document_name:
            return


        self.removeDocumentWorker.emit(
            document_name
        )


    # ========================================================
    # Open source
    # ========================================================

    @Slot(str, str)
    def openSource(
        self,
        document_name,
        page,
    ):

        path = (
            PROJECT_ROOT
            / "documents"
            / document_name
        )


        if not path.exists():
            return


        try:

            page_number = int(
                page
            )

        except (
            ValueError,
            TypeError,
        ):

            page_number = 1


        url = QUrl.fromLocalFile(
            str(path)
        )


        if (
            path.suffix.lower()
            == ".pdf"
        ):

            url.setFragment(
                f"page={page_number}"
            )


        QDesktopServices.openUrl(
            url
        )


    # ========================================================
    # Copy chat
    # ========================================================

    @Slot(str)
    def copyChat(
        self,
        text,
    ):

        QGuiApplication.clipboard().setText(
            text
        )


    # ========================================================
    # CLEAR CHAT
    # ========================================================

    @Slot()
    def clearChat(self):
        """
        Clear only the current UI conversation.

        This does NOT:
            - delete documents
            - delete Qdrant vectors
            - remove indexed data
            - stop Ollama
            - change the RAG prompt
        """

        # ----------------------------------------------------
        # Clear source-related panels
        # ----------------------------------------------------

        self.sourcesChanged.emit(
            "[]"
        )

        self.imagesChanged.emit(
            "[]"
        )

        self.tablesChanged.emit(
            "[]"
        )


        # ----------------------------------------------------
        # Tell QML to clear chat model
        # ----------------------------------------------------

        self.chatCleared.emit()


    # ========================================================
    # Shutdown
    # ========================================================

    @Slot()
    def shutdown(self):

        self.shutdownWorker.emit()


# ============================================================
# Main
# ============================================================

def main():

    app = QGuiApplication(
        sys.argv
    )


    app.setApplicationName(
        "RAG Local Knowledge Assistant"
    )


    if ICON_FILE.exists():

        app.setWindowIcon(
            QIcon(
                str(ICON_FILE)
            )
        )


    # --------------------------------------------------------
    # Lightweight Qt Quick style
    # --------------------------------------------------------

    QQuickStyle.setStyle(
        "Basic"
    )


    # --------------------------------------------------------
    # Worker thread
    # --------------------------------------------------------

    thread = QThread()

    worker = RAGWorker()

    worker.moveToThread(
        thread
    )


    # --------------------------------------------------------
    # UI controller
    # --------------------------------------------------------

    controller = RAGController(
        worker
    )


    thread.finished.connect(
        worker.deleteLater
    )


    # --------------------------------------------------------
    # QML
    # --------------------------------------------------------

    engine = (
        QQmlApplicationEngine()
    )


    engine.rootContext().setContextProperty(
        "ragController",
        controller,
    )


    engine.load(
        str(QML_FILE)
    )


    if not engine.rootObjects():

        thread.quit()

        thread.wait()

        return 1


    # --------------------------------------------------------
    # Start worker thread
    # --------------------------------------------------------

    thread.start()

    controller.initialize()


    # --------------------------------------------------------
    # Run application
    # --------------------------------------------------------

    try:

        exit_code = app.exec()

    finally:

        controller.shutdown()

        thread.quit()

        thread.wait()


    return exit_code


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )