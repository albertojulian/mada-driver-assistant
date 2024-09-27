package com.example.accelgyro

import android.util.Log
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString

class MacWebSocketListener () : WebSocketListener() {

    override fun onOpen(webSocket: WebSocket, response: Response) {
        super.onOpen(webSocket, response)
        // WebSocket abierto, puedes enviar mensajes aquí si lo deseas
        val andr2MacTxt = "Android phone is connected"
        sendMessage(webSocket, andr2MacTxt)
        Log.e("Alberto", "connected")
    }

    override fun onMessage(webSocket: WebSocket, text: String) {

        super.onMessage(webSocket, text)
        // Recibiste un mensaje de tu Mac, puedes manejarlo aquí
    }

    override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
        super.onMessage(webSocket, bytes)
        // Recibiste un mensaje de bytes de tu Mac, puedes manejarlo aquí
    }

    override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
        super.onClosing(webSocket, code, reason)
        webSocket.close(NORMAL_CLOSURE_STATUS, null)
        Log.d("WebSocket", "Closing: $code / $reason")
    }

    override fun onFailure(
        webSocket: WebSocket,
        t: Throwable,
        response: Response?
    ) {
        super.onFailure(webSocket, t, response)
        // Manejar errores de conexión aquí
        Log.d("WebSocket", "Error: " + t.message)
    }

    fun sendMessage(webSocket: WebSocket, message: String) {
        webSocket.send(message)
    }

    companion object {
        private const val NORMAL_CLOSURE_STATUS = 1000
    }
}