package serializer

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"os"
)

func Serialize(filename string, data any) error {
	//bytesData, err := json.Marshal(data)
	//if err != nil {
	//	return err
	//}
	//
	//fmt.Println(len(bytesData))
	modelFile, err := os.Create(fmt.Sprintf("./saved_models/%v.gob", filename))
	if err != nil {
		return err
	}
	encoder := gob.NewEncoder(&bytes.Buffer{})
	defer modelFile.Close()

	err = encoder.Encode(data)
	if err != nil {
		return err
	}

	return nil
}

func Deserialize[T any](filename string, data *T) error {
	modelFile, err := os.Open(fmt.Sprintf("./saved_models/%v.gob", filename))
	if err != nil {
		panic(err)
	}
	defer modelFile.Close()

	dataDecoder := gob.NewDecoder(modelFile)
	err = dataDecoder.Decode(&data)

	if err != nil {
		return err
	}

	return nil
}
