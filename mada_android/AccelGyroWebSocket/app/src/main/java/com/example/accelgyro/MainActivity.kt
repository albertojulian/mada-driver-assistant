package com.example.accelgyro

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.TextView

// Web Sockets management
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.WebSocket

// Accel and gyro
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager

class MainActivity : AppCompatActivity(), SensorEventListener {

    private val serverIP = "192.168.43.233" // dirección IP del servidor en wifi Samsung
    private val serverPort = 8765 // puerto del servidor

    private lateinit var wsListener: MacWebSocketListener
    private lateinit var ws: WebSocket

    // Accel and Gyro part
    private lateinit var wsListener2: MacWebSocketListener
    private lateinit var ws2: WebSocket

    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null

    private lateinit var accelTextView: TextView
    private lateinit var gyroTextView: TextView

    // Variables para controlar el tiempo entre actualizaciones de sensores accel y gyro
    private var lastUpdateTime: Long = 0
    private val updateInterval = 500 // 200 ms = 0.2 segundos

    override fun onCreate(savedInstanceState: Bundle?) {

        val macURL = "ws://" + serverIP + ":$serverPort"
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // web sockets and listeners part

        // accel
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

        // gyro
        Thread {
            try {
                val client = OkHttpClient()
                val request = Request.Builder()
                    .url(macURL) // Dirección IP del Mac en wi-fi AndroidAJR
                    .build()
                wsListener2 = MacWebSocketListener()
                ws2 = client.newWebSocket(request, wsListener)

            } catch (e: Exception) {
                e.printStackTrace()
            }
        }.start()

        // Accel and Gyro part
        accelTextView = findViewById(R.id.accelTextView)
        gyroTextView = findViewById(R.id.gyroTextView)

        // get SensorManager
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager

        // Register accelerometer
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        accelerometer?.also { acc ->
            sensorManager.registerListener(this, acc, SensorManager.SENSOR_DELAY_NORMAL)
        }

        // Register gyroscope
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        gyroscope?.also { gyro ->
            sensorManager.registerListener(this, gyro, SensorManager.SENSOR_DELAY_NORMAL)
        }

    }

    override fun onResume() {
        super.onResume()

        // Accel and gyro
        accelerometer?.also { acc ->
            sensorManager.registerListener(this, acc, SensorManager.SENSOR_DELAY_NORMAL)
        }
        gyroscope?.also { gyro ->
            sensorManager.registerListener(this, gyro, SensorManager.SENSOR_DELAY_NORMAL)
        }
    }

    override fun onPause() {
        super.onPause()

        // accel and gyro
        sensorManager.unregisterListener(this)
    }

    // Accel and gyro methods

    // Método que se ejecuta cuando hay cambios en los sensores
    override fun onSensorChanged(event: SensorEvent?) {
        event?.let {
            // Verificar si han pasado al menos 200 ms desde la última actualización
            val currentTime = System.currentTimeMillis()
            if (currentTime - lastUpdateTime >= updateInterval) {
                lastUpdateTime = currentTime

                when (event.sensor.type) {
                    Sensor.TYPE_ACCELEROMETER -> {
                        val x = event.values[0]
                        val y = event.values[1]
                        val z = event.values[2]
                        accelTextView.text = "Accelerometer\nX: $x\nY: $y\nZ: $z"
                        val accelText = "$x;$y;$z"
                        val wsText = "setAccel " + accelText
                        try {
                            wsListener.sendMessage(ws, wsText)
                        } catch (e: Exception) {
                            e.printStackTrace()
                        }
                    }

                    Sensor.TYPE_GYROSCOPE -> {
                        val x = event.values[0]
                        val y = event.values[1]
                        val z = event.values[2]
                        gyroTextView.text = "Gyroscope\nX: $x\nY: $y\nZ: $z"
                        val gyroText = "$x;$y;$z"
                        val wsText = "setGyro " + gyroText
                        try {
                            wsListener2.sendMessage(ws2, wsText)
                        } catch (e: Exception) {
                            e.printStackTrace()
                        }
                    }
                }
            }
        }
    }

    // Método requerido para liberar los recursos del listener cuando la actividad se detiene
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

}
