package com.example.boonganapps

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.boonganapps.databinding.ActivityMainBinding
import com.example.boonganapps.ml.Model
import com.example.boonganapps.utils.rotateFile
import com.example.boonganapps.utils.uriToFile
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding

    private var getFile: File? = null
    private val imgSize = 640

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

        val image = binding.previewImageView
        binding.apply {
            cameraXButton.setOnClickListener { startCameraX() }
            galleryButton.setOnClickListener { startGallery() }
            predictionButton.setOnClickListener {
                val bitmap = imageViewToBitmap(image) // Convert ImageView to Bitmap
                doInference(bitmap)
            }
        }
    }

    fun imageViewToBitmap(imageView: ImageView): Bitmap {
        imageView.isDrawingCacheEnabled = true
        val bitmap = Bitmap.createBitmap(imageView.drawingCache)
        imageView.isDrawingCacheEnabled = false
        return bitmap
    }

    private fun doInference(bmp: Bitmap) : String{
        val scaledBmp = Bitmap.createScaledBitmap(bmp, imgSize, imgSize, false)
        return outputGenerator(scaledBmp)
    }

    private fun outputGenerator(bitmap: Bitmap):String{
        val model = Model.newInstance(this)


        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 640, 640, 3), DataType.FLOAT32)
        val byteBuffer = ByteBuffer.allocateDirect(4 * imgSize * imgSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(imgSize * imgSize)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        var pixel = 0

        for (i in 0 until imgSize) {
            for (j in 0 until imgSize) {
                val `val` = intValues[pixel++] // RGB
                byteBuffer.putFloat((`val` shr 16 and 0xFF) * (1f / 255))
                byteBuffer.putFloat((`val` shr 8 and 0xFF) * (1f / 255))
                byteBuffer.putFloat((`val` and 0xFF) * (1f / 255))
            }
        }
        inputFeature0.loadBuffer(byteBuffer)

    // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
            .outputAsCategoryList
//
//        val output = outputs[0]

//        Log.e("INIIIIIIIIIII",outputs.toString())
        binding.textView.text = outputs.toString()

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
                val bitmap = BitmapFactory.decodeStream(contentResolver.openInputStream(uri))
                val myFile = uriToFile(uri, this@MainActivity)
                getFile = myFile
                binding.previewImageView.setImageURI(uri)
                outputGenerator(bitmap)
            }
        }
    }

    companion object {
        const val CAMERA_X_RESULT = 200

        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        private const val REQUEST_CODE_PERMISSIONS = 10
    }
}