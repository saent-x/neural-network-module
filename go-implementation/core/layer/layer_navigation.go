package layer

type LayerNavigation struct {
	prev interface{}
	next interface{}
}

func (self *LayerNavigation) GetPreviousLayer() interface{} {
	return self.prev
}

func (self *LayerNavigation) GetNextLayer() interface{} {
	return self.next
}

func (self *LayerNavigation) SetPreviousLayer(prev interface{}) {
	self.prev = prev
}

func (self *LayerNavigation) SetNextLayer(next interface{}) {
	self.next = next
}
