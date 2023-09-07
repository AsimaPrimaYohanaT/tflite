package com.example.boonganapps

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.net.Uri
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.boonganapps.databinding.ActivityMainBinding
import com.example.boonganapps.ml.Detect
import com.example.boonganapps.utils.rotateFile
import com.example.boonganapps.utils.uriToFile
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private var getFile: File? = null

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (!allPermissionsGranted()) {
                Toast.makeText(
                    this,
                    "Tidak mendapatkan permission.",
                    Toast.LENGTH_SHORT
                ).show()
                finish()
            }
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(
                this,
                REQUIRED_PERMISSIONS,
                REQUEST_CODE_PERMISSIONS
            )
        }


        binding.apply {
            cameraXButton.setOnClickListener { startCameraX() }
            galleryButton.setOnClickListener { startGallery() }
            predictionButton.setOnClickListener {
                predict()
            }
        }
    }
    private fun predict() {
        val input = binding.previewImageView

        val model = Detect.newInstance(this)

        val modelInputSize = 320

        val labels =  application.assets.open("label.txt").bufferedReader().readLines()
        val inputBuffer = imageViewToByteBuffer(input, modelInputSize)

        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, modelInputSize, modelInputSize, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(inputBuffer)

        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
        val outputFeature1 = outputs.outputFeature1AsTensorBuffer
        val outputFeature2 = outputs.outputFeature2AsTensorBuffer
        val outputFeature3 = outputs.outputFeature3AsTensorBuffer

        val confidence = outputFeature0.floatArray
        var maxIdx = 0
        outputFeature0.floatArray.forEachIndexed { idx, fl ->
            if (fl > outputFeature3.floatArray[maxIdx]) {
                maxIdx = idx
            }
        }

        var maxPos = 0
        var maxConfidence = -10f
        for (i in confidence.indices) {
            if (confidence[i] > maxConfidence) {
                maxConfidence = confidence[i]
                maxPos = i
            }
        }
        val resultConfidence = confidence[0]
        val formattedConfidence = String.format("%.2f", resultConfidence)
        name = labels[maxIdx]
        binding.textView.text = "nama: ${name}, confidence: ${formattedConfidence}"

    }

    private fun imageViewToByteBuffer(imageView: ImageView, modelInputSize: Int): ByteBuffer {

        val bitmap = imageViewToBitmap(imageView)
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, modelInputSize, modelInputSize, true)
        val byteBuffer = ByteBuffer.allocateDirect(1 * modelInputSize * modelInputSize * 3 * 4) // 3 channels (RGB) * 4 bytes per float
        byteBuffer.order(java.nio.ByteOrder.nativeOrder())
        val intValues = IntArray(modelInputSize * modelInputSize)
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)

        var pixel = 0
        for (i in 0 until modelInputSize) {
            for (j in 0 until modelInputSize) {
                val value = intValues[pixel++]
                byteBuffer.putFloat(((value shr 16 and 0xFF) - 127.5f) / 127.5f)
                byteBuffer.putFloat(((value shr 8 and 0xFF) - 127.5f) / 127.5f)
                byteBuffer.putFloat(((value and 0xFF) - 127.5f) / 127.5f)
            }
        }
        byteBuffer.rewind()

        return byteBuffer
    }

    private fun imageViewToBitmap(imageView: ImageView): Bitmap {
        val drawable = imageView.drawable
        val bitmap = Bitmap.createBitmap(imageView.width, imageView.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        drawable?.setBounds(0, 0, canvas.width, canvas.height)
        drawable?.draw(canvas)
        return bitmap
    }


    private fun startCameraX() {
        val intent = Intent(this, CameraActivity::class.java)
        launcherIntentCameraX.launch(intent)
    }

    private fun startGallery() {
        val intent = Intent()
        intent.action = Intent.ACTION_GET_CONTENT
        intent.type = "image/*"
        val chooser = Intent.createChooser(intent, "Choose a Picture")
        launcherIntentGallery.launch(chooser)
    }

    private val launcherIntentCameraX = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) {
        if (it.resultCode == CAMERA_X_RESULT) {
            val myFile = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                it.data?.getSerializableExtra("picture", File::class.java)
            } else {
                @Suppress("DEPRECATION")
                it.data?.getSerializableExtra("picture")
            } as? File

            val isBackCamera = it.data?.getBooleanExtra("isBackCamera", true) as Boolean

            myFile?.let { file ->
                rotateFile(file, isBackCamera)
                getFile = file
                binding.previewImageView.setImageBitmap(BitmapFactory.decodeFile(file.path))
            }
        }
    }
    private val launcherIntentGallery = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            val selectedImg = result.data?.data as Uri
            selectedImg.let { uri ->
                val myFile = uriToFile(uri, this@MainActivity)
                getFile = myFile
                binding.previewImageView.setImageURI(uri)
            }
        }
    }

    companion object {
        const val CAMERA_X_RESULT = 200
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        private const val REQUEST_CODE_PERMISSIONS = 10
        var name:String? = null
    }
}