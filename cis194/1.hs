---- https://www.seas.upenn.edu/~cis194/spring13/lectures.html

---- Exercise 1

toDigits :: Integer -> [Integer]
toDigits n 
  | n > 0 = foldr (\x acc -> (read [x] :: Integer) : acc) [] (show n)
  | otherwise = []

toDigitsRev :: Integer -> [Integer]
toDigitsRev n 
  | n > 0 =  foldl (\acc x -> (read [x] :: Integer) : acc) [] (show n)
  | otherwise = []

-- helper to debugging types
-- mapResult = map (\x -> read [x] :: Integer) "112"

---- Exercise 2

doubleEveryOther :: [Integer] -> [Integer]
doubleEveryOther = reverse . removeIndex . map doubleIndexEven . addIndex . reverse

-- helpers for doubleEveryOther
addIndex :: [Integer] -> [(Integer, Integer)]
addIndex = zip [1..]

isIndexEven :: (Integer, Integer) -> Bool 
isIndexEven (i, n) 
  | i `mod` 2 == 0 = True
  | otherwise = False

doubleIndexEven :: (Integer, Integer) -> (Integer, Integer) 
doubleIndexEven (i, n) 
  | i `mod` 2 == 0 = (i, n*2)
  | otherwise = (i, n)

removeIndex :: [(Integer, Integer)] -> [Integer]
removeIndex = map snd

---- Exercise 3

-- example: sumDigits [16,7,12,5] = 1 + 6 + 7 + 1 + 2 + 5 = 22
sumDigits :: [Integer] -> Integer 
sumDigits = sum . convertToSingleDigits

-- helper
convertToSingleDigits :: [Integer] -> [Integer]
convertToSingleDigits = concat . map toDigits

---- Exercise 4

-- example: validate 4012888888881881 = True
-- example: validate 4012888888881882 = False
validate :: Integer -> Bool
validate i
  | checksum i `mod` 10 == 0 = True
  | otherwise = False

checksum :: Integer -> Integer
checksum = sumDigits . doubleEveryOther . toDigits
