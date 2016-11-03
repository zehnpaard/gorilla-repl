;; gorilla-repl.fileformat = 1

;; **
;;; # 線形回帰を最急降下法で解く２
;;; 
;;; 
;;; 今回はclojure.core.matrixの標準、jblasを使ったvectorz-clj、そしてblasを直接使うclatrixの三種類のcore.matrix実装を試す。
;;; 
;;; (X.T)Xが特異行列で逆行列が求められないことがあったため、前回同様、数値解の求め方である最急降下法で解く。
;; **

;; @@
(ns indigo-leaves
  (:refer-clojure :exclude [+ - * / == < <= > >= not= = min max])
  (:require 
    [clojure.core.matrix :as m]
    [clojure.core.matrix.operators :refer :all]
    [clojure.core.matrix.random :as mr]
    [gorilla-plot.core :as plot]))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"}
;; <=

;; @@
;(m/set-current-implementation :ndarray)
(m/set-current-implementation :vectorz)
;(m/set-current-implementation :clatrix)
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-keyword'>:vectorz</span>","value":":vectorz"}
;; <=

;; **
;;; まずはデータ作成。
;;; 
;;; x値を0~1の連続一様分布からとり、y値を以下の式から求める：
;;; 
;;; $$y=2+5x_1+3x_2+0.5\epsilon$$
;;; 
;;; @@\epsilon@@はガウス分布に従う確率変数とする。
;; **

;; @@
(def n 50)
(def theta (m/matrix [[2.0 5.0 3.0]]))
(def sigma 0.50)

(def x0 (m/matrix (repeat n 1)))
(def x1 (mr/sample-uniform n))
(def x2 (mr/sample-uniform n))

(def X
  (m/join
    (m/reshape x0 [1 n])
    (m/reshape x1 [1 n])
    (m/reshape x2 [1 n])))

(def y-without-error (m/mmul theta X))

(def errors (* (mr/sample-normal n) sigma))

(def y (+ y-without-error errors))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;indigo-leaves/y</span>","value":"#'indigo-leaves/y"}
;; <=

;; **
;;; $$\hat{y} = \theta_0 + \theta_1 x$$
;;; とした時に、@@\hat{y}@@と@@y@@の差の二乗を最小に留める@@\theta=[\theta_0,\theta_1]@@を求めたい。
;;; 
;;; @@J@@はその差の二乗をコスト関数として表したもの。
;; **

;; @@
(defn J [theta X y]
  (-> (m/mmul theta X) 
      (- y) 
      (** 2) 
      m/esum 
      (/ (* 2 (m/ecount y)))))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;indigo-leaves/J</span>","value":"#'indigo-leaves/J"}
;; <=

;; **
;;; 最急降下法は@@\theta@@の添字@@j@@につき
;;; 
;;; $$\theta_j := \theta_j - \alpha\frac{1}{m} \sum^m_i(h_t(x^{(i)}) - y^{(i)})x^{(i)}_j$$
;;; 
;;; （ただし@@\sum^m\_i@@は@@\sum^m\_{i=1}@@、そして@@h\_t@@は@@h\_{\theta}@@）
;;; 
;;; （@@\alpha@@は一回の更新でどれだけ大きく@@\theta@@を変化させるかをコントロールする学習レート変数）
;;; 
;;; の更新をすべての@@j@@で_同時_に行うことを繰り返して最適解に近似していく。
;;; 
;;; 
;; **

;; @@
(defn next-theta [theta X y alpha]
  (let [z (m/ecount y)
        hx (m/mmul theta X)
        hx-y (- hx y)
        hx-yx (* (m/reshape hx-y [z]) X)
        s (apply map + (m/transpose hx-yx))
        t (/ s z)
        u (* t alpha)]
    (- theta u)))

(defn theta-after-n-iteration [n]
  (let [learning-rate 0.01
        f #(next-theta % X y learning-rate)
        initial-theta (m/matrix (repeat (m/row-count X) 1.0))]
    (->> initial-theta
        (iterate f)
        (drop n)
        first)))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;indigo-leaves/theta-after-n-iteration</span>","value":"#'indigo-leaves/theta-after-n-iteration"}
;; <=

;; **
;;; @@\theta_{15000}@@：15000回繰り返した結果
;; **

;; @@
(time
  (doall (theta-after-n-iteration 15000)))
;; @@
;; ->
;;; &quot;Elapsed time: 1133.936101 msecs&quot;
;;; 
;; <-
;; =>
;;; {"type":"html","content":"<span class='clj-unkown'>#vectorz/vector [1.8850786173294962,4.97670451141408,2.885097397286794]</span>","value":"#vectorz/vector [1.8850786173294962,4.97670451141408,2.885097397286794]"}
;; <=

;; **
;;; @@\theta_{30000}@@：30000回繰り返した結果
;; **

;; @@
(time 
  (doall (theta-after-n-iteration 30000)))
;; @@
;; ->
;;; &quot;Elapsed time: 2122.331857 msecs&quot;
;;; 
;; <-
;; =>
;;; {"type":"html","content":"<span class='clj-unkown'>#vectorz/vector [1.881619902630865,4.979437022948518,2.8889278718090208]</span>","value":"#vectorz/vector [1.881619902630865,4.979437022948518,2.8889278718090208]"}
;; <=

;; **
;;; @@\theta\_{15000}@@と@@\theta_{30000}@@の差がわずかであり、さらに実際の分布の母数である2.0と5.0を近似していることがわかる。
;;; 
;;; また、途中経過の@@\theta@@と@@J(\theta)@@を表示する関数二つ。コスト@@J@@が減少していくのがわかる。
;; **

;; @@
(defn intermediate-thetas [n interval]
  (let [learning-rate 0.01
        f #(next-theta % X y learning-rate)
        initial-theta (m/matrix (repeat (m/row-count X) 1.0))]
    (->> initial-theta
         (iterate f)
         (take (inc n))
         (take-nth interval))))

(time
  (def thetas (doall (intermediate-thetas 15000 1000))))

thetas
(map #(J % X y) thetas)
;; @@
;; ->
;;; &quot;Elapsed time: 1066.40142 msecs&quot;
;;; 
;; <-
;; =>
;;; {"type":"list-like","open":"<span class='clj-lazy-seq'>(</span>","close":"<span class='clj-lazy-seq'>)</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-double'>8.552340278510624</span>","value":"8.552340278510624"},{"type":"html","content":"<span class='clj-double'>0.167670755477041</span>","value":"0.167670755477041"},{"type":"html","content":"<span class='clj-double'>0.11605237019474395</span>","value":"0.11605237019474395"},{"type":"html","content":"<span class='clj-double'>0.09811034239256286</span>","value":"0.09811034239256286"},{"type":"html","content":"<span class='clj-double'>0.09100076308257986</span>","value":"0.09100076308257986"},{"type":"html","content":"<span class='clj-double'>0.0879913404513631</span>","value":"0.0879913404513631"},{"type":"html","content":"<span class='clj-double'>0.08668035251476157</span>","value":"0.08668035251476157"},{"type":"html","content":"<span class='clj-double'>0.08610253773598385</span>","value":"0.08610253773598385"},{"type":"html","content":"<span class='clj-double'>0.08584668812612548</span>","value":"0.08584668812612548"},{"type":"html","content":"<span class='clj-double'>0.08573319618150127</span>","value":"0.08573319618150127"},{"type":"html","content":"<span class='clj-double'>0.0856828170748049</span>","value":"0.0856828170748049"},{"type":"html","content":"<span class='clj-double'>0.08566044767023186</span>","value":"0.08566044767023186"},{"type":"html","content":"<span class='clj-double'>0.08565051412207794</span>","value":"0.08565051412207794"},{"type":"html","content":"<span class='clj-double'>0.08564610276475991</span>","value":"0.08564610276475991"},{"type":"html","content":"<span class='clj-double'>0.08564414370807542</span>","value":"0.08564414370807542"},{"type":"html","content":"<span class='clj-double'>0.08564327369765003</span>","value":"0.08564327369765003"}],"value":"(8.552340278510624 0.167670755477041 0.11605237019474395 0.09811034239256286 0.09100076308257986 0.0879913404513631 0.08668035251476157 0.08610253773598385 0.08584668812612548 0.08573319618150127 0.0856828170748049 0.08566044767023186 0.08565051412207794 0.08564610276475991 0.08564414370807542 0.08564327369765003)"}
;; <=

;; @@

;; @@
