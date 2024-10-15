package main

import (
	"encoding/hex"
	"fmt"
	"log"
	"math"
	"strconv"
)

func main() {
	flValue, err := strconv.ParseInt("19D", 16, 64)
	if err != nil {
		log.Fatalf("Error converting hex to decimal: %v", err)
	}
	fmt.Println("19D -> ", float64(flValue))

	flValue2, err := strconv.ParseUint("C0003FFD000000FF", 16, 64)
	if err != nil {
		log.Fatalf("Error converting hex to decimal: %v", err)
	}

	fmt.Println("C0003FFD000000FF -> ", float64(flValue2))
	//for _, hex := range hexValues {
	//	float32Value, float64Value := HexToFloat(hex)
	//	fmt.Printf("Hex: %s\n", hex)
	//	fmt.Printf("  As float32: %f\n", float32Value)
	//	fmt.Printf("  As float64: %f\n\n", float64Value)
	//}
}

func hexToDecimal(hexValues []string) []float64 {
	decimalValues := make([]float64, len(hexValues))
	for i, hex := range hexValues {
		decValue, err := strconv.ParseInt(hex, 16, 64)
		if err != nil {
			log.Fatalf("Error converting hex to decimal: %v", err)
		}
		decimalValues[i] = float64(decValue)
	}
	return decimalValues
}

func HexToFloat(hexStr string) (float32, float64) {
	// Pad the hex string to ensure it's 8 or 16 characters long
	for len(hexStr) < 8 {
		hexStr = "0" + hexStr
	}

	// Decode the hex string to bytes
	bytes, err := hex.DecodeString(hexStr)
	if err != nil {
		panic(err)
	}

	// Reverse the byte order (Go uses little-endian, but the input is big-endian)
	for i := 0; i < len(bytes)/2; i++ {
		bytes[i], bytes[len(bytes)-1-i] = bytes[len(bytes)-1-i], bytes[i]
	}

	// Convert to float32
	var float32Value float32
	if len(bytes) >= 4 {
		float32Value = math.Float32frombits(uint32(bytes[0]) | uint32(bytes[1])<<8 | uint32(bytes[2])<<16 | uint32(bytes[3])<<24)
	}

	// Convert to float64
	var float64Value float64
	if len(bytes) == 8 {
		float64Value = math.Float64frombits(uint64(bytes[0]) | uint64(bytes[1])<<8 | uint64(bytes[2])<<16 | uint64(bytes[3])<<24 |
			uint64(bytes[4])<<32 | uint64(bytes[5])<<40 | uint64(bytes[6])<<48 | uint64(bytes[7])<<56)
	}

	return float32Value, float64Value
}
