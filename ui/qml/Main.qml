import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

ApplicationWindow {
    id: window

    visible: true
    width: 1280
    height: 800
    minimumWidth: 900
    minimumHeight: 600

    title: "RAG It"
    color: "#202B22"

    // ========================================================
    // Theme
    // ========================================================

    property color olive: "#202B22"
    property color panel: "#28342A"
    property color panelLight: "#314035"
    property color yellow: "#FFD85F"
    property color white: "#F8F8F5"
    property color muted: "#B8C0B9"
    property color black: "#0D0F0E"
    property color border: "#3A463C"
    property color danger: "#E05B5B"

    property bool documentsVisible: true
    property bool sourcesVisible: true
    property bool ragPromptVisible: false
    property bool ready: false

    // ========================================================
    // Models
    // ========================================================

    ListModel {
        id: chatModel
    }

    ListModel {
        id: documentModel
    }

    ListModel {
        id: sourceModel
    }

    ListModel {
        id: imageModel
    }

    ListModel {
        id: tableModel
    }


    // ========================================================
    // Add Document Dialog
    // ========================================================

    FileDialog {
        id: addDocumentDialog

        title: "Add Document"

        nameFilters: [
            "Documents (*.pdf *.docx *.pptx *.xlsx *.html *.md *.txt)"
        ]

        onAccepted: {
            ragController.addDocument(
                selectedFile.toString()
            )
        }
    }


    // ========================================================
    // Clear Chat Confirmation
    // ========================================================

    Dialog {
        id: clearChatDialog

        title: "Clear Chat"
        modal: true

        anchors.centerIn: parent

        width: 360

        standardButtons:
            Dialog.Yes | Dialog.No

        Text {
            width: parent.width

            text:
                "Clear the current conversation and retrieved sources?\n\n"
                + "Your documents and Qdrant vectors will not be deleted."

            color: window.white
            font.pixelSize: 13
            wrapMode: Text.Wrap

            padding: 10
        }

        onAccepted: {
            ragController.clearChat()
        }
    }


    // ========================================================
    // Backend connections
    // ========================================================

    Connections {
        target: ragController


        function onReady() {

            window.ready = true

            statusText.text = "Ready"

            statusDot.color =
                window.yellow
        }


        function onStatusChanged(message) {

            statusText.text = message

            if (message === "Ready") {

                statusDot.color =
                    window.yellow

            } else if (message === "Error") {

                statusDot.color =
                    window.danger

            } else {

                statusDot.color =
                    "#D6B94A"
            }
        }


        function onErrorOccurred(message) {

            statusText.text = "Error"

            chatModel.append({
                role: "assistant",
                text: "Error: " + message
            })

            chatList.positionViewAtEnd()
        }


        // ====================================================
        // CLEAR CHAT
        // ====================================================

        function onChatCleared() {

            chatModel.clear()

            sourceModel.clear()

            imageModel.clear()

            tableModel.clear()

            questionInput.clear()

            statusText.text = "Ready"

            statusDot.color =
                window.yellow

            chatList.positionViewAtBeginning()
        }


        // ====================================================
        // Documents
        // ====================================================

        function onDocumentsChanged(data) {

            documentModel.clear()

            var documents =
                JSON.parse(data)

            for (
                var i = 0;
                i < documents.length;
                ++i
            ) {

                documentModel.append({
                    name: documents[i]
                })
            }

            documentList.currentIndex = -1
        }


        // ====================================================
        // Sources
        // ====================================================

        function onSourcesChanged(data) {

            sourceModel.clear()

            var sources =
                JSON.parse(data)

            for (
                var i = 0;
                i < sources.length;
                ++i
            ) {

                sourceModel.append({

                    document:
                        sources[i].document
                        || "Unknown",

                    page:
                        String(
                            sources[i].page
                            || ""
                        ),

                    type:
                        sources[i].type
                        || "unknown",

                    score:
                        String(
                            sources[i].score
                            || ""
                        ),

                    chunkId:
                        sources[i].chunk_id
                        || "",

                    content:
                        sources[i].content
                        || "",

                    assetPath:
                        sources[i].asset_path
                        || ""
                })
            }
        }


        // ====================================================
        // Images
        // ====================================================

        function onImagesChanged(data) {

            imageModel.clear()

            var images =
                JSON.parse(data)

            for (
                var i = 0;
                i < images.length;
                ++i
            ) {

                imageModel.append({

                    path:
                        images[i].path
                        || "",

                    document:
                        images[i].document
                        || "",

                    page:
                        String(
                            images[i].page
                            || ""
                        ),

                    score:
                        String(
                            images[i].score
                            || ""
                        )
                })
            }
        }


        // ====================================================
        // Tables
        // ====================================================

        function onTablesChanged(data) {

            tableModel.clear()

            var tables =
                JSON.parse(data)

            for (
                var i = 0;
                i < tables.length;
                ++i
            ) {

                tableModel.append({

                    document:
                        tables[i].document
                        || "",

                    page:
                        String(
                            tables[i].page
                            || ""
                        ),

                    content:
                        tables[i].content
                        || "",

                    score:
                        String(
                            tables[i].score
                            || ""
                        )
                })
            }
        }


        // ====================================================
        // Answer
        // ====================================================

        function onAnswerStarted() {

            chatModel.append({
                role: "assistant",
                text: ""
            })

            chatList.positionViewAtEnd()
        }


        function onAnswerToken(token) {

            if (
                chatModel.count === 0
            ) {
                return
            }

            var index =
                chatModel.count - 1

            var current =
                chatModel.get(index)

            chatModel.setProperty(
                index,
                "text",
                current.text + token
            )

            chatList.positionViewAtEnd()
        }


        function onAnswerFinished() {

            statusText.text =
                "Ready"

            statusDot.color =
                window.yellow
        }
    }


    // ========================================================
    // Helpers
    // ========================================================

    function sendQuestion() {

        if (!window.ready) {
            return
        }

        var question =
            questionInput.text.trim()

        if (!question) {
            return
        }

        chatModel.append({
            role: "user",
            text: question
        })

        sourceModel.clear()

        imageModel.clear()

        tableModel.clear()

        ragController.ask(
            question,
            ragPromptInput.text
        )

        questionInput.clear()

        chatList.positionViewAtEnd()
    }


    function copyAllChat() {

        var output = ""

        for (
            var i = 0;
            i < chatModel.count;
            ++i
        ) {

            var item =
                chatModel.get(i)

            output +=
                item.role === "user"
                ? "YOU:\n"
                : "RAG:\n"

            output += item.text

            output += "\n\n"
        }

        ragController.copyChat(
            output.trim()
        )
    }


    function copyMessage(text) {

        ragController.copyChat(
            text
        )
    }


    // ========================================================
    // Root
    // ========================================================

    Rectangle {

        anchors.fill: parent

        color: window.olive


        // ====================================================
        // Header
        // ====================================================

        Rectangle {

            id: header

            anchors.top: parent.top
            anchors.left: parent.left
            anchors.right: parent.right

            height: 58

            color: window.black

            border.color:
                window.border

            border.width: 1


            RowLayout {

                anchors.fill: parent

                anchors.leftMargin: 14
                anchors.rightMargin: 14

                spacing: 8


                // =================================================
                // Documents panel toggle
                // =================================================

                Rectangle {

                    width: 32
                    height: 30

                    radius: 6

                    color:
                        window.panel


                    Text {

                        anchors.centerIn:
                            parent

                        text:
                            window.documentsVisible
                            ? "◀"
                            : "▶"

                        color:
                            window.yellow

                        font.pixelSize: 13
                    }


                    MouseArea {

                        anchors.fill:
                            parent

                        cursorShape:
                            Qt.PointingHandCursor

                        onClicked: {

                            window.documentsVisible =
                                !window.documentsVisible
                        }
                    }
                }


                // =================================================
                // Sources panel toggle
                // =================================================

                Rectangle {

                    width: 32
                    height: 30

                    radius: 6

                    color:
                        window.panel


                    Text {

                        anchors.centerIn:
                            parent

                        text:
                            window.sourcesVisible
                            ? "▶"
                            : "◀"

                        color:
                            window.yellow

                        font.pixelSize: 13
                    }


                    MouseArea {

                        anchors.fill:
                            parent

                        cursorShape:
                            Qt.PointingHandCursor

                        onClicked: {

                            window.sourcesVisible =
                                !window.sourcesVisible
                        }
                    }
                }


                // =================================================
                // Title
                // =================================================

                Text {

                    text:
                        "RAG It"

                    color:
                        window.white

                    font.pixelSize: 16

                    font.bold: true

                    Layout.leftMargin: 6

                    Layout.alignment: Qt.AlignBaseline
                }

                Text {

                    text:
                        "Local Knowledge Assistent"

                    color:
                        window.white

                    font.pixelSize: 12

                    font.bold: false

                    Layout.leftMargin: 6

                    Layout.alignment: Qt.AlignBaseline
                }


                Item {
                    Layout.fillWidth: true
                }


                // =================================================
                // CLEAR CHAT
                // =================================================

                Rectangle {

                    width: 92
                    height: 30

                    radius: 6

                    color:
                        window.panel


                    border.color:
                        window.border

                    border.width: 1


                    Text {

                        anchors.centerIn:
                            parent

                        text:
                            "CLEAR CHAT"

                        color:
                            window.yellow

                        font.pixelSize: 10

                        font.bold: true
                    }


                    MouseArea {

                        anchors.fill:
                            parent

                        cursorShape:
                            Qt.PointingHandCursor

                        onClicked: {

                            if (
                                chatModel.count > 0
                                || sourceModel.count > 0
                                || imageModel.count > 0
                                || tableModel.count > 0
                            ) {

                                clearChatDialog.open()
                            }
                        }
                    }
                }


                // =================================================
                // COPY ALL
                // =================================================

                Rectangle {

                    width: 58
                    height: 30

                    radius: 6

                    color:
                        window.panel


                    Text {

                        anchors.centerIn:
                            parent

                        text:
                            "COPY"

                        color:
                            window.yellow

                        font.pixelSize: 10

                        font.bold: true
                    }


                    MouseArea {

                        anchors.fill:
                            parent

                        cursorShape:
                            Qt.PointingHandCursor

                        onClicked:
                            copyAllChat()
                    }
                }


                // =================================================
                // Status
                // =================================================

                Rectangle {

                    id: statusDot

                    width: 9
                    height: 9

                    radius: 5

                    color:
                        window.yellow

                    Layout.leftMargin: 8
                }


                Text {

                    id: statusText

                    text:
                        "Starting..."

                    color:
                        window.white

                    font.pixelSize: 12
                }
            }
        }


        // ====================================================
        // Main layout
        // ====================================================

        RowLayout {

            anchors.top:
                header.bottom

            anchors.bottom:
                parent.bottom

            anchors.left:
                parent.left

            anchors.right:
                parent.right

            spacing: 0


            // =================================================
            // Documents
            // =================================================

            Rectangle {

                visible:
                    window.documentsVisible

                Layout.preferredWidth:
                    235

                Layout.fillHeight:
                    true

                color:
                    window.panel

                border.color:
                    window.border

                border.width:
                    1


                ColumnLayout {

                    anchors.fill:
                        parent

                    anchors.margins:
                        14

                    spacing:
                        10


                    RowLayout {

                        Layout.fillWidth:
                            true

                        spacing:
                            6


                        Text {

                            text:
                                "DOCUMENTS"

                            color:
                                window.white

                            font.pixelSize:
                                12

                            font.bold:
                                true
                        }


                        Item {
                            Layout.fillWidth:
                                true
                        }


                        // =================================================
                        // Add document
                        // =================================================

                        Rectangle {

                            width: 28
                            height: 28

                            radius: 6

                            color:
                                window.panelLight


                            Text {

                                anchors.centerIn:
                                    parent

                                text:
                                    "+"

                                color:
                                    window.yellow

                                font.pixelSize:
                                    18

                                font.bold:
                                    true
                            }


                            MouseArea {

                                anchors.fill:
                                    parent

                                cursorShape:
                                    Qt.PointingHandCursor

                                onClicked: {

                                    addDocumentDialog.open()
                                }
                            }
                        }


                        // =================================================
                        // Remove document
                        // =================================================

                        Rectangle {

                            width: 28
                            height: 28

                            radius: 6

                            color:
                                documentList.currentIndex >= 0
                                ? window.panelLight
                                : "#333A35"


                            Text {

                                anchors.centerIn:
                                    parent

                                text:
                                    "−"

                                color:
                                    documentList.currentIndex >= 0
                                    ? window.yellow
                                    : window.muted

                                font.pixelSize:
                                    18

                                font.bold:
                                    true
                            }


                            MouseArea {

                                anchors.fill:
                                    parent

                                cursorShape:
                                    Qt.PointingHandCursor


                                onClicked: {

                                    if (
                                        documentList.currentIndex >= 0
                                    ) {

                                        ragController.removeDocument(

                                            documentModel.get(
                                                documentList.currentIndex
                                            ).name
                                        )

                                        documentList.currentIndex =
                                            -1
                                    }
                                }
                            }
                        }
                    }


                    Rectangle {

                        Layout.fillWidth:
                            true

                        height:
                            1

                        color:
                            window.border
                    }


                    ListView {

                        id:
                            documentList

                        Layout.fillWidth:
                            true

                        Layout.fillHeight:
                            true

                        clip:
                            true

                        model:
                            documentModel

                        currentIndex:
                            -1

                        spacing:
                            3


                        delegate:
                            Rectangle {

                                width:
                                    documentList.width

                                height:
                                    42

                                radius:
                                    6


                                color:
                                    documentList.currentIndex === index
                                    ? "#3A4A3D"
                                    : (
                                        index % 2 === 0
                                        ? "#263129"
                                        : "transparent"
                                    )


                                RowLayout {

                                    anchors.fill:
                                        parent

                                    anchors.leftMargin:
                                        10

                                    anchors.rightMargin:
                                        8

                                    spacing:
                                        8


                                    Text {

                                        text:
                                            "▣"

                                        color:
                                            window.yellow

                                        font.pixelSize:
                                            13
                                    }


                                    Text {

                                        text:
                                            name

                                        color:
                                            window.white

                                        font.pixelSize:
                                            11

                                        elide:
                                            Text.ElideMiddle

                                        Layout.fillWidth:
                                            true
                                    }
                                }


                                MouseArea {

                                    anchors.fill:
                                        parent

                                    cursorShape:
                                        Qt.PointingHandCursor

                                    onClicked: {

                                        documentList.currentIndex =
                                            index
                                    }
                                }
                            }
                    }


                    Rectangle {

                        Layout.fillWidth:
                            true

                        height:
                            38

                        radius:
                            7

                        color:
                            window.black

                        border.color:
                            window.border

                        border.width:
                            1


                        Text {

                            anchors.centerIn:
                                parent

                            text:
                                "Open Documents Folder"

                            color:
                                window.muted

                            font.pixelSize:
                                11
                        }


                        MouseArea {

                            anchors.fill:
                                parent

                            cursorShape:
                                Qt.PointingHandCursor

                            onClicked: {

                                Qt.openUrlExternally(
                                    ragController.documentsPath
                                )
                            }
                        }
                    }
                }
            }


            // =================================================
            // Chat
            // =================================================

            Rectangle {

                Layout.fillWidth:
                    true

                Layout.fillHeight:
                    true

                color:
                    window.olive


                ColumnLayout {

                    anchors.fill:
                        parent

                    spacing:
                        0


                    ListView {

                        id:
                            chatList

                        Layout.fillWidth:
                            true

                        Layout.fillHeight:
                            true

                        clip:
                            true

                        spacing:
                            12

                        leftMargin:
                            18

                        rightMargin:
                            18

                        topMargin:
                            16

                        bottomMargin:
                            10

                        model:
                            chatModel


                        onCountChanged: {

                            Qt.callLater(
                                function() {

                                    chatList.positionViewAtEnd()
                                }
                            )
                        }


                        delegate:
                            Item {

                                width:
                                    chatList.width

                                height:
                                    bubble.height + 4


                                Rectangle {

                                    id:
                                        bubble

                                    width:
                                        Math.min(
                                            chatList.width - 36,
                                            760
                                        )

                                    height:
                                        body.implicitHeight + 28

                                    radius:
                                        10


                                    color:
                                        role === "user"
                                        ? window.yellow
                                        : window.panel


                                    border.color:
                                        role === "user"
                                        ? "transparent"
                                        : window.border


                                    anchors.right:
                                        role === "user"
                                        ? parent.right
                                        : undefined

                                    anchors.left:
                                        role === "assistant"
                                        ? parent.left
                                        : undefined


                                    anchors.rightMargin:
                                        role === "user"
                                        ? 18
                                        : 0

                                    anchors.leftMargin:
                                        role === "assistant"
                                        ? 18
                                        : 0


                                    // =================================================
                                    // Copy message
                                    // =================================================

                                    Rectangle {

                                        width:
                                            28

                                        height:
                                            28

                                        radius:
                                            6

                                        anchors.right:
                                            parent.right

                                        anchors.top:
                                            parent.top

                                        anchors.rightMargin:
                                            8

                                        anchors.topMargin:
                                            8

                                        color:
                                            role === "user"
                                            ? "#E7C454"
                                            : window.panelLight


                                        Text {

                                            anchors.centerIn:
                                                parent

                                            text:
                                                "⧉"

                                            color:
                                                role === "user"
                                                ? window.black
                                                : window.white

                                            font.pixelSize:
                                                14
                                        }


                                        MouseArea {

                                            anchors.fill:
                                                parent

                                            cursorShape:
                                                Qt.PointingHandCursor

                                            onClicked:
                                                copyMessage(
                                                    model.text
                                                )
                                        }
                                    }


                                    Column {

                                        id:
                                            body

                                        anchors.left:
                                            parent.left

                                        anchors.right:
                                            parent.right

                                        anchors.top:
                                            parent.top

                                        anchors.margins:
                                            14

                                        spacing:
                                            8


                                        Text {

                                            text:
                                                role === "user"
                                                ? "YOU"
                                                : "RAG"

                                            color:
                                                role === "user"
                                                ? window.black
                                                : window.yellow

                                            font.pixelSize:
                                                10

                                            font.bold:
                                                true

                                            font.letterSpacing:
                                                1
                                        }


                                        Text {

                                            text:
                                                model.text

                                            textFormat:
                                                role === "assistant"
                                                ? Text.MarkdownText
                                                : Text.PlainText

                                            color:
                                                role === "user"
                                                ? window.black
                                                : window.white

                                            font.pixelSize:
                                                14

                                            wrapMode:
                                                Text.Wrap

                                            width:
                                                bubble.width - 52

                                            linkColor:
                                                window.yellow


                                            onLinkActivated:
                                                function(link) {

                                                    Qt.openUrlExternally(
                                                        link
                                                    )
                                                }
                                        }
                                    }
                                }
                            }


                        Text {

                            visible:
                                chatModel.count === 0

                            anchors.centerIn:
                                parent

                            text:
                                "Ask a question about your documents"

                            color:
                                window.muted

                            font.pixelSize:
                                20
                        }
                    }


                    // =================================================
                    // Retrieved images
                    // =================================================

                    Rectangle {

                        visible:
                            imageModel.count > 0

                        Layout.fillWidth:
                            true

                        Layout.preferredHeight:
                            imageModel.count > 0
                            ? 132
                            : 0

                        color:
                            window.panel

                        border.color:
                            window.border

                        border.width:
                            1


                        ColumnLayout {

                            anchors.fill:
                                parent

                            anchors.margins:
                                10

                            spacing:
                                6


                            Text {

                                text:
                                    "RELATED IMAGES"

                                color:
                                    window.yellow

                                font.pixelSize:
                                    10

                                font.bold:
                                    true
                            }


                            ListView {

                                Layout.fillWidth:
                                    true

                                Layout.fillHeight:
                                    true

                                orientation:
                                    ListView.Horizontal

                                spacing:
                                    8

                                clip:
                                    true

                                model:
                                    imageModel


                                delegate:
                                    Rectangle {

                                        width:
                                            150

                                        height:
                                            96

                                        radius:
                                            7

                                        color:
                                            window.black

                                        border.color:
                                            window.border

                                        border.width:
                                            1


                                        Image {

                                            anchors.fill:
                                                parent

                                            anchors.margins:
                                                5

                                            source:
                                                path

                                            fillMode:
                                                Image.PreserveAspectFit

                                            asynchronous:
                                                true

                                            cache:
                                                true
                                        }


                                        MouseArea {

                                            anchors.fill:
                                                parent

                                            cursorShape:
                                                Qt.PointingHandCursor

                                            onClicked: {

                                                Qt.openUrlExternally(
                                                    path
                                                )
                                            }
                                        }
                                    }
                            }
                        }
                    }


                    // =================================================
                    // Retrieved tables
                    // =================================================

                    Rectangle {

                        visible:
                            tableModel.count > 0

                        Layout.fillWidth:
                            true

                        Layout.preferredHeight:
                            tableModel.count > 0
                            ? 160
                            : 0

                        color:
                            window.panel

                        border.color:
                            window.border

                        border.width:
                            1


                        ColumnLayout {

                            anchors.fill:
                                parent

                            anchors.margins:
                                10

                            spacing:
                                6


                            Text {

                                text:
                                    "RELATED TABLES"

                                color:
                                    window.yellow

                                font.pixelSize:
                                    10

                                font.bold:
                                    true
                            }


                            ListView {

                                Layout.fillWidth:
                                    true

                                Layout.fillHeight:
                                    true

                                clip:
                                    true

                                spacing:
                                    8

                                model:
                                    tableModel


                                delegate:
                                    Rectangle {

                                        width:
                                            ListView.view.width

                                        height:
                                            118

                                        radius:
                                            7

                                        color:
                                            window.black

                                        border.color:
                                            window.border

                                        border.width:
                                            1


                                        Flickable {

                                            anchors.fill:
                                                parent

                                            anchors.margins:
                                                8

                                            contentWidth:
                                                tableText.paintedWidth

                                            contentHeight:
                                                tableText.paintedHeight

                                            clip:
                                                true


                                            Text {

                                                id:
                                                    tableText

                                                width:
                                                    Math.max(
                                                        parent.width,
                                                        480
                                                    )

                                                text:
                                                    content

                                                color:
                                                    window.white

                                                font.family:
                                                    "Consolas"

                                                font.pixelSize:
                                                    11

                                                wrapMode:
                                                    Text.NoWrap
                                            }
                                        }
                                    }
                            }
                        }
                    }


                    // =================================================
                    // RAG prompt
                    // =================================================

                    Rectangle {

                        Layout.fillWidth:
                            true

                        Layout.preferredHeight:
                            window.ragPromptVisible
                            ? 150
                            : 42

                        color:
                            window.panel

                        border.color:
                            window.border

                        border.width:
                            1


                        ColumnLayout {

                            anchors.fill:
                                parent

                            spacing:
                                0


                            Text {

                                Layout.fillWidth:
                                    true

                                Layout.preferredHeight:
                                    38

                                leftPadding:
                                    18

                                topPadding:
                                    10

                                text:
                                    window.ragPromptVisible
                                    ? "RAG PROMPT ▲"
                                    : "RAG PROMPT ▼"

                                color:
                                    window.yellow

                                font.pixelSize:
                                    11

                                font.bold:
                                    true


                                MouseArea {

                                    anchors.fill:
                                        parent

                                    cursorShape:
                                        Qt.PointingHandCursor

                                    onClicked: {

                                        window.ragPromptVisible =
                                            !window.ragPromptVisible
                                    }
                                }
                            }


                            TextArea {

                                id:
                                    ragPromptInput

                                visible:
                                    window.ragPromptVisible

                                Layout.fillWidth:
                                    true

                                Layout.fillHeight:
                                    true

                                Layout.leftMargin:
                                    18

                                Layout.rightMargin:
                                    18

                                Layout.bottomMargin:
                                    10


                                text:
                                    ragController.defaultRagPrompt

                                color:
                                    window.white

                                font.pixelSize:
                                    12

                                wrapMode:
                                    TextArea.Wrap

                                placeholderText:
                                    "Instructions specific to this RAG..."

                                placeholderTextColor:
                                    window.muted


                                background:
                                    Rectangle {

                                        color:
                                            window.olive

                                        radius:
                                            6

                                        border.color:
                                            ragPromptInput.activeFocus
                                            ? window.yellow
                                            : window.border

                                        border.width:
                                            1
                                    }
                            }
                        }
                    }


                    // =================================================
                    // Input
                    // =================================================

                    Rectangle {

                        Layout.fillWidth:
                            true

                        height:
                            72

                        color:
                            window.panel

                        border.color:
                            window.border

                        border.width:
                            1


                        RowLayout {

                            anchors.fill:
                                parent

                            anchors.leftMargin:
                                18

                            anchors.rightMargin:
                                18

                            anchors.topMargin:
                                10

                            anchors.bottomMargin:
                                10

                            spacing:
                                10


                            TextArea {

                                id:
                                    questionInput

                                Layout.fillWidth:
                                    true

                                Layout.fillHeight:
                                    true


                                color:
                                    window.white

                                font.pixelSize:
                                    14

                                wrapMode:
                                    TextArea.Wrap

                                placeholderText:
                                    "Ask something about your documents..."

                                placeholderTextColor:
                                    window.muted


                                background:
                                    Rectangle {

                                        color:
                                            window.olive

                                        radius:
                                            8

                                        border.color:
                                            questionInput.activeFocus
                                            ? window.yellow
                                            : window.border

                                        border.width:
                                            1
                                    }


                                Keys.onPressed:
                                    function(event) {

                                        if (
                                            event.key === Qt.Key_Return
                                            ||
                                            event.key === Qt.Key_Enter
                                        ) {

                                            if (
                                                event.modifiers
                                                &
                                                Qt.ShiftModifier
                                            ) {

                                                return
                                            }


                                            event.accepted =
                                                true

                                            sendQuestion()
                                        }
                                    }
                            }


                            // =================================================
                            // Send
                            // =================================================

                            Rectangle {

                                id:
                                    sendButton

                                Layout.preferredWidth:
                                    76

                                Layout.fillHeight:
                                    true

                                radius:
                                    8


                                color:
                                    window.ready
                                    ? window.yellow
                                    : "#555C56"


                                Text {

                                    anchors.centerIn:
                                        parent

                                    text:
                                        "SEND"

                                    color:
                                        window.ready
                                        ? window.black
                                        : window.muted

                                    font.pixelSize:
                                        11

                                    font.bold:
                                        true
                                }


                                MouseArea {

                                    anchors.fill:
                                        parent

                                    cursorShape:
                                        Qt.PointingHandCursor

                                    onClicked:
                                        sendQuestion()
                                }
                            }
                        }
                    }
                }
            }


            // =================================================
            // Sources
            // =================================================

            Rectangle {

                visible:
                    window.sourcesVisible

                Layout.preferredWidth:
                    280

                Layout.fillHeight:
                    true

                color:
                    window.panel

                border.color:
                    window.border

                border.width:
                    1


                ColumnLayout {

                    anchors.fill:
                        parent

                    anchors.margins:
                        14

                    spacing:
                        10


                    RowLayout {

                        Layout.fillWidth:
                            true


                        Text {

                            text:
                                "SOURCES"

                            color:
                                window.white

                            font.pixelSize:
                                12

                            font.bold:
                                true
                        }


                        Item {
                            Layout.fillWidth:
                                true
                        }


                        Text {

                            text:
                                sourceModel.count

                            color:
                                window.yellow

                            font.pixelSize:
                                12

                            font.bold:
                                true
                        }
                    }


                    Rectangle {

                        Layout.fillWidth:
                            true

                        height:
                            1

                        color:
                            window.border
                    }


                    ListView {

                        id:
                            sourceList

                        Layout.fillWidth:
                            true

                        Layout.fillHeight:
                            true

                        clip:
                            true

                        spacing:
                            8

                        model:
                            sourceModel


                        delegate:
                            Rectangle {

                                width:
                                    sourceList.width

                                height:
                                    88

                                radius:
                                    8

                                color:
                                    window.black

                                border.color:
                                    window.border

                                border.width:
                                    1


                                Column {

                                    anchors.fill:
                                        parent

                                    anchors.margins:
                                        10

                                    spacing:
                                        5


                                    Text {

                                        text:
                                            document

                                        color:
                                            window.yellow

                                        font.pixelSize:
                                            11

                                        font.bold:
                                            true

                                        width:
                                            parent.width

                                        elide:
                                            Text.ElideMiddle
                                    }


                                    Text {

                                        text:
                                            "Page "
                                            + page
                                            + "  •  "
                                            + type

                                        color:
                                            window.muted

                                        font.pixelSize:
                                            10
                                    }


                                    Text {

                                        text:
                                            "Score "
                                            + score

                                        color:
                                            window.yellow

                                        font.pixelSize:
                                            10
                                    }


                                    Text {

                                        text:
                                            chunkId

                                        color:
                                            "#7F8B82"

                                        font.pixelSize:
                                            9

                                        width:
                                            parent.width

                                        elide:
                                            Text.ElideMiddle
                                    }
                                }


                                MouseArea {

                                    anchors.fill:
                                        parent

                                    cursorShape:
                                        Qt.PointingHandCursor

                                    onClicked: {

                                        ragController.openSource(
                                            document,
                                            page
                                        )
                                    }
                                }
                            }


                        Text {

                            visible:
                                sourceModel.count === 0

                            anchors.centerIn:
                                parent

                            text:
                                "Sources will appear here"

                            color:
                                window.muted

                            font.pixelSize:
                                11
                        }
                    }
                }
            }
        }
    }
}