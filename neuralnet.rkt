;; The first three lines of this file were inserted by DrRacket. They record metadata
;; about the language level of this file in a form that our tools can easily process.
#reader(lib "htdp-intermediate-lambda-reader.ss" "lang")((modname neuralnet) (read-case-sensitive #t) (teachpacks ((lib "universe.rkt" "teachpack" "2htdp") (lib "image.rkt" "teachpack" "2htdp"))) (htdp-settings #(#t constructor repeating-decimal #f #t none #f ((lib "universe.rkt" "teachpack" "2htdp") (lib "image.rkt" "teachpack" "2htdp")) #f)))
;(require plot)
;(require clotho)

#;(parameterize ([current-random-source (make-random-source #f)])
(println (random -1 1)))

(define (rand _)
  (local [(define decimal (/ (random 1000) 1000))]
    (if (= (random 2) 1) (* -1 decimal) decimal)))
;-------------------------------
; HELPER FUNCTIONS
;-------------------------------
; sigma : [List-of Num] -> Num
(define (sigma lon)
  (foldr + 0 lon))

;-------------------------------
; PARAMETERS
;-------------------------------
; a Layer is a [List-of Num]

(define NUMLAYERS 2)
(define NUMFEATURES 1)
(define NUMNODES-L1 10)
(define NUMNODES-L2 1)
(define MODEL-STRUCTURE (list NUMFEATURES NUMNODES-L1 NUMNODES-L2))

; a Parameters is a [List-of [List-of [List-of Num]]]
; first index is layer index in the network l
; second index is the next-node index j
; third index is the previous-node weight index i
; for a Layer of length n, each [List-of Num] has length n+1

; init-parameters : [List-of Nat] -> [List-of [List-of [List-of Num]]]
; the list should have the size of each layer in nodes, sequentially
(define (init-parameters list)
  (local [; create-lon : Nat Nat -> [List-of Num]
          ; creates a list of random nums, of a given length
          (define (create-lon i length)
            (if (= i length) empty
                (cons (rand 0) (create-lon (add1 i) length))))
          ; create-lolon : Nat Nat Nat -> [List-of [List-of Num]]
          (define (create-lolon j length height)
            (if (= j height) empty
                (cons (create-lon 0 length) (create-lolon (add1 j) length height))))]
    (cond
      [(empty? (rest list)) '()]
      [(cons? (rest list)) (cons (create-lolon 0 (add1 (first list)) (second list))
                                 (init-parameters (rest list)))])))

(define PARAMETERS1 (init-parameters MODEL-STRUCTURE))
(define PARAMETERS2 (init-parameters MODEL-STRUCTURE))
#|
   #prev node 1
[([w1 ... b] #next node 1
      ...      #layer 1
  [w1 ... b]) 

      ...

 ([w1 ... b]
      ...      #layer n
  [w1 ... b])]

|#

  


;-------------------------------
; DATASET
;-------------------------------
; an Example is a (make-posn x y)
; a Dataset is a [List-of Example]
(define NUMEXAMPLES 100)
(define FUNCTION sin)

; make-example : Num [Num -> Num] -> Posn
(define (make-example x f)
  (make-posn x (f x)))

; make-domain : Num Num Nat -> [List-of Num]
(define (make-domain min step numpoints)
  (if (= 0 numpoints) empty (cons min (make-domain (+ min step) step (sub1 numpoints)))))

; make-dataset : [List-of Num] [Num -> Num] -> [List-of Posn]
(define (make-dataset domain func)
  (map [lambda (x) (make-example x func)] domain))

(define DOMAIN (make-domain 0 1 NUMEXAMPLES))
(define DATASET (make-dataset DOMAIN FUNCTION))
;-------------------------------
; MODEL
;-------------------------------
; model : Example -> Num
(define (model x params)
  (local
    [; ReLU : Num -> Num
     (define (ReLU x)
       (max x 0))
     ; compute-linear : Layer [List-of Num] -> Num 
     (define (compute-linear xs weights)
       (foldr + 0
              (map *
                   (append xs (list 1))
                   weights)))
     
     ; compute-node : Natural Natural Layer -> Num
     (define (compute-node l-index j-index layer activation)
       (activation (compute-linear layer (list-ref (list-ref params l-index) j-index))))
     ; compute-layer : Natural Natural Layer -> Layer
     (define (compute-layer l-index j-index layer activation)
       (if (= (length (list-ref params l-index)) j-index)
           empty
           (cons (compute-node l-index j-index layer activation) (compute-layer l-index (add1 j-index) layer activation))))]
    
    ;(compute-layer 0 0 x)))
    (compute-layer 1 0 (compute-layer 0 0 x ReLU) (lambda (x) x))))

;-------------------------------
; LOSS FUNCTION
;-------------------------------

; MSEloss : Parameters -> Num
(define (MSEloss params)
  (local
    [; sqr-error : Posn -> Num
     (define (sqr-error example)
       (square (- (posn-y example) (model (posn-x example) params))))]
     
    (/ (sigma (map sqr-error DATASET) NUMEXAMPLES))))


;-------------------------------
; GRADIENT-DESCENT
;-------------------------------
(define ALPHA 1)

#;(define (gradient-descent model param)
    (local [; back-prop: Model Parameters -> Parameters
            ; update-params: Parameters Parameters -> Parameters
            (deep-map + param (deep-map * ALPHA gradients))]))

(define L1 (list 1 1 (list (list 1)) (list 1 1) (list (list 1) (list 1 1))))
(define L2 (list 1 2 (list (list 3)) (list 4 5) (list (list 6) (list 7 8))))

; deep-map : [List-of X] [List-of Y]
(check-expect (deep-map + L1 L2) (list 2 3
                                       (list
                                        (list 4))
                                       (list 5 6)
                                       (list
                                        (list 7)
                                        (list 8 9))))
(define (deep-map op l1 l2)
  (cond
    [(and (empty? l1) (empty? l2)) empty]
    [(and (number? (first l2)) (number? (first l1))) (cons (op (first l1) (first l2)) (deep-map op (rest l1) (rest l2)))]
    [(and (cons? (first l1)) (cons? (first l2))) (cons
                                                  (deep-map op (first l1) (first l2))
                                                  (deep-map op (rest l1) (rest l2)))]))

;-------------------------------
; TRAINING AND GRAPHING
;-------------------------------
(define THRESHOLD 0.001)

(define POINT (circle 0.2 "solid" "red"))
(define POINT2 (circle 0.2 "solid" "blue"))

(define PLT-WIDTH 80)
(define PLT-HEIGHT 40)
;(define ORIGIN (make-posn (/ PLT-WIDTH 2) (/ PLT-HEIGHT 2)))
(define ORIGIN (make-posn 0 (/ PLT-HEIGHT 2)))

(define BG (empty-scene PLT-WIDTH PLT-HEIGHT))

; compute : [List-of Posn] -> [List-of Posn]
(define (compute dataset params)
  (local [; point-compute : Posn -> Posn
          (define (point-compute p)
            (make-posn (posn-x p) (first (model (list (posn-x p)) params))))]
    (map point-compute dataset)))

; get-point : Posn -> Posn
(define (get-point p)
  (make-posn (+ (posn-x p) (posn-x ORIGIN)) (- PLT-HEIGHT (+ (posn-y p) (posn-y ORIGIN)))))


; optimized? : Parameters -> Boolean
(define (optimized? params)
  (<= (MSEloss params) THRESHOLD))

; gradient-descent : Parameters Model Dataset -> Parameters
(define (gradient-descent _)
  (map (λ (param) 0)))

; plot-graph : Parameters -> Image
(define (plot-graph params)
  (scale 10
         (place-images
          (make-list NUMEXAMPLES POINT2)
          (map get-point (compute DATASET params))
          (place-images
              (make-list NUMEXAMPLES POINT)
              (map get-point DATASET)
              BG))))

(define (train-model-main params)
  (big-bang params
    [on-tick gradient-descent]
    [to-draw plot-graph]
    [stop-when optimized?]))