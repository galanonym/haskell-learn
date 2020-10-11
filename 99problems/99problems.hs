-- 1 Last element of the list

myLast :: [a] -> a
myLast [] = error "Not for empty lists"
myLast [x] = x
myLast (_:xs) = myLast xs

-- 2 Last but one

lastButOne :: [a] -> a
lastButOne [] = error "Not for empty lists"
lastButOne (x:[_]) = x
lastButOne (_:xs) = lastButOne xs

-- 3 Find K'th element

elementAt :: Int -> [a] -> a
elementAt k xs = xs !! (k-1) 

elementAt' :: Int -> [a] -> a
elementAt' 1 (x:_) = x
elementAt' _ [] = error "Out of bound"
elementAt' k (_:xs)
  | k < 1 = error "Out of bound 2"
  | otherwise = elementAt' (k - 1) xs

-- 4 Find number of elements

myLength :: [a] -> Int
myLength [] = 0
myLength (_:xs) = 1 + myLength xs

myLengthPf :: [a] -> Int
myLengthPf = fst. last . zip [1..]
