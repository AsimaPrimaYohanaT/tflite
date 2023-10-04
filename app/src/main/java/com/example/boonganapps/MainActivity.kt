package com.example.boonganapps

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.graphics.drawable.BitmapDrawable
import android.net.Uri
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.boonganapps.databinding.ActivityMainBinding
import com.example.boonganapps.ml.DetectMD
import com.example.boonganapps.ml.ModelNew
import com.example.boonganapps.utils.rotateFile
import com.example.boonganapps.utils.uriToFile
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
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

        val desiredWidth = 320
        val desiredHeight = 320

        val layoutParams = ViewGroup.LayoutParams(desiredWidth, desiredHeight)

        input.layoutParams = layoutParams

        input.scaleType = ImageView.ScaleType.FIT_CENTER

        val bitmap: Bitmap = (input.drawable as BitmapDrawable).bitmap
        val model = DetectMD.newInstance(this)

        val grayscaleBitmap = convertToGrayscale(bitmap, desiredWidth, desiredHeight)

        val resizedBitmap = Bitmap.createScaledBitmap(grayscaleBitmap, desiredWidth, desiredHeight, true)

        val image = TensorImage.fromBitmap(resizedBitmap)
        val outputs = model.process(image)

        val imageWithBoundingBox = resizedBitmap.copy(Bitmap.Config.ARGB_8888, true)

        val paint = Paint().apply {
            color = Color.RED
            style = Paint.Style.STROKE
            strokeWidth = 4.0f
            textSize = 90f
        }

        val canvas = Canvas(imageWithBoundingBox)
        val objectInfoList = mutableListOf<Pair<Float, String>>()


        for (i in 0 until outputs.detectionResultList.size) {
            val obj = outputs.detectionResultList[i]
            val conf = obj.scoreAsFloat
            val bbox = obj.locationAsRectF
            val label = obj.categoryAsString

            var minConf = 0.5f

            if (conf > minConf) {
                canvas.drawRect(bbox, paint)
                canvas.drawText("$label: %.2f".format(conf), bbox.left, bbox.top - 10, paint)
                val objectInfo = Pair(bbox.right, "$label: %.2f".format(conf))
                objectInfoList.add(objectInfo)
            }
        }

        objectInfoList.sortByDescending { it.first }

        val labelsAndConf = objectInfoList.joinToString(", ") { it.second }
        binding.textView.text = labelsAndConf
        binding.previewImageView.setImageBitmap(imageWithBoundingBox)
        model.close()
    }

    fun convertToGrayscale(inputBitmap: Bitmap, targetWidth: Int, targetHeight: Int): Bitmap {
        val width = inputBitmap.width
        val height = inputBitmap.height
        val grayscaleBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(grayscaleBitmap)
        val paint = Paint()
        val colorMatrix = ColorMatrix()
        colorMatrix.setSaturation(0f) // Set saturation to 0 for grayscale
        val colorMatrixFilter = ColorMatrixColorFilter(colorMatrix)
        paint.colorFilter = colorMatrixFilter
        canvas.drawBitmap(inputBitmap, 0f, 0f, paint)
        return grayscaleBitmap
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