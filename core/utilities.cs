using NumSharp;
using NumSharp.Generic;

namespace IDS_NN.core;

public static class @Utilities
{
	public static void PrintArray<T>(this T[,] array)
	{
		for (int i = 0; i < array.GetLength(0); i++)
		{
			for (int j = 0; j < array.GetLength(1); j++)
			{
				Console.Write(array[i,j] + "\t");
			}
			Console.WriteLine();
		}
	}
	public static double[,] ConcatenateColwise(params double[][] arrs)
	{
		// each arr must have same length else throw exception
		if (arrs.Any(arr => arr.Length != arrs[0].Length))
			throw new ArgumentException("arrays must have the same length");
		
		var noOfCols = arrs.Length;
		var noOfRows = arrs[0].Length;
		
		var concatenatedArrs = new double[noOfRows, noOfCols];

		for (var i = 0; i < noOfCols; i++)
			for (var j = 0; j < noOfRows; j++)
				concatenatedArrs[j,i] = arrs[i][j];

		return concatenatedArrs;
	}
	public static (NDArray, NDArray) SpiralData(int points, int classes)
	{
		var X = np.zeros((points * classes, 2));
		var y = np.zeros((Shape) (points * classes), dtype: np.uint8);
		
		for (int class_number = 0; class_number < Enumerable.Range(0, classes).Count(); class_number++)
		{
			// define index range
			// NDArray ix = Enumerable.Range(points * class_number, points * (class_number + 1))
			//  		.ToArray();
			
			var ix = np.arange(points * class_number, points * (class_number + 1));

			// define radius and theta
			var r = np.linspace(0.0, 1, points);
			var t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2;
			
			var a = r * np.sin(t * 2.5);
			var b = r * np.cos(t * 2.5);
			
			//Console.WriteLine($"r: {r.max()} t: {t.max()} | sin_a: {a.max()}, cos_b: {b.max()}");

			var e = ConcatenateColwise((double[])a, (double[])b);
			//e.PrintArray();
			
			X[ix] = e;
			
			//Console.WriteLine($"[{h}]");
			
			// assign class label
			y[ix] = class_number;
		}
		
		return (X,y);
	}
} 