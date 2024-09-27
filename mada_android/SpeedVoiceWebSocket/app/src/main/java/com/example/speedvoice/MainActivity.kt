package com.example.speedvoice

import android.Manifest
import android.content.pm.PackageManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.TextView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import okhttp3.OkHttpClient
import okhttp3.Request
import com.google.android.gms.location.*
import okhttp3.WebSocket
import kotlin.math.roundToInt

import android.content.Intent
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.RecognitionListener
// import androidx.activity.result.contract.ActivityResultContracts
import java.util.*
import android.util.Log

class MainActivity : AppCompatActivity() {

    private val serverIP = "192.168.43.233" // dirección IP del servidor en wifi Samsung
    private val serverPort = 8765 // puerto del servidor

    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private lateinit var locationRequest: LocationRequest
    private lateinit var locationCallback: LocationCallback

    private lateinit var wsListener: MacWebSocketListener
    private lateinit var ws: WebSocket

    private lateinit var speedTextView: TextView

    var speedText = "1000"

    private lateinit var voiceTextView: TextView
    private lateinit var speechRecognizer: SpeechRecognizer
    private var listenMode = false

    override fun onCreate(savedInstanceState: Bundle?) {

        val macURL = "ws://" + serverIP + ":$serverPort"
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Speed and web sockets part
        speedTextView = findViewById(R.id.speedTextView)
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        createLocationRequest()

        Thread {
            try {
                val client = OkHttpClient()
                val request = Request.Builder()
                    .url(macURL) // Dirección IP del Mac en wi-fi AndroidAJR
                    .build()
                wsListener = MacWebSocketListener()
                ws = client.newWebSocket(request, wsListener)

            } catch (e: Exception) {
                e.printStackTrace()
            }
        }.start()

        locationCallback = object : LocationCallback() {

            override fun onLocationResult(locationResult: LocationResult?) {
                locationResult ?: return
                for (location in locationResult.locations) {
                    val speed = (location.speed * 3.6).roundToInt() // m/s a km/h

                    speedText = "$speed" // websocket uses this parameter
                    val wsText = "setSpeed " + speedText
                    wsListener.sendMessage(ws, wsText)
                    speedTextView.text = speedText + " km/h" // update screen
                }
            }
        }

        // Voice Recognition part
        voiceTextView = findViewById(R.id.voiceTextView)

        // Solicitar permiso en tiempo de ejecución
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(android.Manifest.permission.RECORD_AUDIO), 101)
        } else {
            initializeSpeechRecognizer()
        }
    }

    private fun initializeSpeechRecognizer() {
        // Inicializar el SpeechRecognizer
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        // Configurar el listener para el reconocimiento de voz
        speechRecognizer.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                Log.d("SpeechRecognizer", "Ready for speech")
            }

            override fun onBeginningOfSpeech() {
                Log.d("SpeechRecognizer", "Speech beginning")
            }

            override fun onRmsChanged(rmsdB: Float) {
                // Puede ser útil para mostrar niveles de volumen de la voz, pero aquí lo ignoramos
            }

            override fun onBufferReceived(buffer: ByteArray?) {}

            override fun onEndOfSpeech() {
                Log.d("SpeechRecognizer", "Speech ended")
            }

            override fun onResults(results: Bundle?) {
                val recognizedText = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.get(0)
                recognizedText?.let {
                    if (it == "listen") {
                        listenMode = true
                        sendSpeechText(it)
                    } else if (listenMode == true) {
                        listenMode = false
                        sendSpeechText(it)
                    } else {
                        voiceTextView.text = "Say \"listen\" before asking something"
                    }

                }
                // Reiniciar la escucha después de obtener resultados
                startVoiceRecognition()
            }

            override fun onError(error: Int) {
                Log.e("SpeechRecognizer", "Error: $error")
                if (error == SpeechRecognizer.ERROR_NO_MATCH) {
                    startVoiceRecognition() // Reiniciar si no se reconoció la voz
                } else if (error == SpeechRecognizer.ERROR_SPEECH_TIMEOUT) {
                    startVoiceRecognition() // Reiniciar si timeout sin actividad
                } else if (error == SpeechRecognizer.ERROR_RECOGNIZER_BUSY) {
                    // Podrías agregar un retraso aquí si el recognizer está ocupado
                    speechRecognizer.cancel()
                    startVoiceRecognition()
                }
            }

            override fun onPartialResults(partialResults: Bundle?) {}

            override fun onEvent(eventType: Int, params: Bundle?) {}

        })

        // Iniciar la escucha
        startVoiceRecognition()

    }

    private fun sendSpeechText(it: String) {

        voiceTextView.text = it
        // Send input message ws to web socket server in Mac
        val wsText = "setInputMessage " + it
        wsListener.sendMessage(ws, wsText)

    }

    private fun createLocationRequest() {
        locationRequest = LocationRequest.create().apply {
            interval = 500  // Intervalo de actualización en milisegundos (1 segundo)
            fastestInterval = 500  // Intervalo más rápido de actualización en milisegundos (1 segundo)
            priority = LocationRequest.PRIORITY_HIGH_ACCURACY
        }
    }

    override fun onResume() {
        super.onResume()

        if (checkLocationPermission()) {
            startLocationUpdates()
        } else {
            requestLocationPermission()
        }
    }

    override fun onPause() {
        super.onPause()
        stopLocationUpdates()
    }

    private fun startLocationUpdates() {
        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            // TODO: Consider calling
            //    ActivityCompat#requestPermissions
            // here to request the missing permissions, and then overriding
            //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
            //                                          int[] grantResults)
            // to handle the case where the user grants the permission. See the documentation
            // for ActivityCompat#requestPermissions for more details.
            return
        }
        fusedLocationClient.requestLocationUpdates(
            locationRequest,
            locationCallback,
            null
        )
    }

    private fun stopLocationUpdates() {
        fusedLocationClient.removeLocationUpdates(locationCallback)
    }

    private fun checkLocationPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            android.Manifest.permission.ACCESS_FINE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestLocationPermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(android.Manifest.permission.ACCESS_FINE_LOCATION),
            LOCATION_PERMISSION_REQUEST_CODE
        )
    }

    companion object {
        private const val LOCATION_PERMISSION_REQUEST_CODE = 1001
    }

    private fun startVoiceRecognition() {
        val speechRecognizerIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            // putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            // putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.US)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US")
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, "en-US") // Preferencia de idioma en inglés
            putExtra(RecognizerIntent.EXTRA_ONLY_RETURN_LANGUAGE_PREFERENCE, "en-US") // Solo devolver resultados en inglés
            // putExtra(RecognizerIntent.EXTRA_PROMPT, "Speak now...")
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, packageName)
        }

        speechRecognizer.startListening(speechRecognizerIntent)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 101 && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            initializeSpeechRecognizer()
        } else {
            // Permiso denegado, maneja la situación aquí
            Log.e("SpeechRecognizer", "SpeechRecognizer not permitted")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Liberar recursos del SpeechRecognizer
        speechRecognizer.destroy()
    }

}
