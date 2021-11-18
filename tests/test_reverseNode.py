import unittest
import src.reverseNode

class ReverseNodeTests(unittest.TestCase):
  
  def test_init(self):
    value = 0.1
    func = src.reverseNode.ReverseNode(value)
    self.assertEqual(func.value, 0.1)
    self.assertEqual(func.adjoint, 1)
    self.assertEqual(func.children, [])

  def test_gradient(self):
    value = 0.1
    x = src.reverseNode.ReverseNode(value)
    self.assertEqual(x.gradient(), 1)
    func = x * src.reverseNode.ReverseNode(2.0)
    func2 = x + src.reverseNode.ReverseNode(1.0)
    self.assertNotEqual(x.children, [])
    self.assertEqual(x.gradient(), 3)

  def test_add_(self):
    value = 0.1
    x = src.reverseNode.ReverseNode(value)
    func = x + src.reverseNode.ReverseNode(2.0)
    self.assertEqual(func.value, 2.1)

  def test_sub_(self):
    value = 0.1
    x = src.reverseNode.ReverseNode(value)
    func = x - src.reverseNode.ReverseNode(2.0)
    self.assertEqual(func.value, -1.9)

  def test_mul_(self):
    value = 0.1
    x = src.reverseNode.ReverseNode(value)
    func = x * src.reverseNode.ReverseNode(2.0)
    self.assertEqual(func.value, 0.2)


  def test_pow_(self):
    value = 2
    x = src.reverseNode.ReverseNode(value)
    func = x ** src.reverseNode.ReverseNode(2.0)
    self.assertEqual(func.value, 4)

if __name__ == "__main__":
  unittest.main()
