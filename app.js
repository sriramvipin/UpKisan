// Global Variables
let latitude, longitude;

// Get User Location
function getLocation() {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(showPosition, showError);
    } else {
        alert("Geolocation is not supported by this browser.");
    }
}

// Show GPS Position
function showPosition(position) {
    latitude = position.coords.latitude;
    longitude = position.coords.longitude;
    document.getElementById("location").innerHTML = `Location: ${latitude}, ${longitude}`;
    fetchNDVIData(latitude, longitude);
}

// Error Handling
function showError(error) {
    switch(error.code) {
        case error.PERMISSION_DENIED:
            alert("User denied the request for Geolocation.");
            break;
        case error.POSITION_UNAVAILABLE:
            alert("Location information is unavailable.");
            break;
        case error.TIMEOUT:
            alert("The request to get user location timed out.");
            break;
        case error.UNKNOWN_ERROR:
            alert("An unknown error occurred.");
            break;
    }
}

// Fetch NDVI Data from API (Placeholder)
async function fetchNDVIData(lat, lon) {
    const apiUrl = `https://api.example.com/ndvi?lat=${lat}&lon=${lon}&key=YOUR_API_KEY`;
    
    try {
        const response = await fetch(apiUrl);
        const data = await response.json();
        
        if (data.ndvi) {
            updateNDVI(data.ndvi);
        } else {
            document.getElementById("ndvi-status").innerHTML = "Unable to fetch NDVI data.";
        }
    } catch (error) {
        console.error("Error fetching NDVI data:", error);
    }
}

// Update NDVI and Suggest Fertilizer
function updateNDVI(ndviValue) {
    let healthStatus = "";
    let suggestion = "";

    if (ndviValue > 0.6) {
        healthStatus = "✅ Healthy";
        suggestion = "Minimal fertilizer required.";
    } else if (ndviValue > 0.3) {
        healthStatus = "⚠️ Moderate Stress";
        suggestion = "Apply nitrogen-based fertilizers.";
    } else {
        healthStatus = "❗ Critical Condition";
        suggestion = "Apply high phosphorus and potassium fertilizers.";
    }

    document.getElementById("ndvi-status").innerHTML = `NDVI Value: ${ndviValue.toFixed(2)} - ${healthStatus}`;
    document.getElementById("vra-suggestion").innerHTML = `VRA Suggestion: ${suggestion}`;
}

// Auto Update NDVI Every 3 Hours
setInterval(() => {
    if (latitude && longitude) {
        fetchNDVIData(latitude, longitude);
    }
}, 3 * 60 * 60 * 1000); // 3 hours in milliseconds
