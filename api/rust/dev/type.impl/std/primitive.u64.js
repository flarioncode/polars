(function() {
    var type_impls = Object.fromEntries([["polars_utils",[["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-AbsDiff-for-u64\" class=\"impl\"><a class=\"src rightside\" href=\"src/polars_utils/abs_diff.rs.html#49\">source</a><a href=\"#impl-AbsDiff-for-u64\" class=\"anchor\">§</a><h3 class=\"code-header\">impl <a class=\"trait\" href=\"polars_utils/abs_diff/trait.AbsDiff.html\" title=\"trait polars_utils::abs_diff::AbsDiff\">AbsDiff</a> for <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u64.html\">u64</a></h3></section></summary><div class=\"impl-items\"><section id=\"associatedtype.Abs\" class=\"associatedtype trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/abs_diff.rs.html#49\">source</a><a href=\"#associatedtype.Abs\" class=\"anchor\">§</a><h4 class=\"code-header\">type <a href=\"polars_utils/abs_diff/trait.AbsDiff.html#associatedtype.Abs\" class=\"associatedtype\">Abs</a> = <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u64.html\">u64</a></h4></section><section id=\"method.max_abs_diff\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/abs_diff.rs.html#49\">source</a><a href=\"#method.max_abs_diff\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/abs_diff/trait.AbsDiff.html#tymethod.max_abs_diff\" class=\"fn\">max_abs_diff</a>() -&gt; Self::<a class=\"associatedtype\" href=\"polars_utils/abs_diff/trait.AbsDiff.html#associatedtype.Abs\" title=\"type polars_utils::abs_diff::AbsDiff::Abs\">Abs</a></h4></section><section id=\"method.abs_diff\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/abs_diff.rs.html#49\">source</a><a href=\"#method.abs_diff\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/abs_diff/trait.AbsDiff.html#tymethod.abs_diff\" class=\"fn\">abs_diff</a>(self, other: Self) -&gt; Self::<a class=\"associatedtype\" href=\"polars_utils/abs_diff/trait.AbsDiff.html#associatedtype.Abs\" title=\"type polars_utils::abs_diff::AbsDiff::Abs\">Abs</a></h4></section></div></details>","AbsDiff","polars_utils::index::IdxSize"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-DirtyHash-for-u64\" class=\"impl\"><a class=\"src rightside\" href=\"src/polars_utils/hashing.rs.html#83\">source</a><a href=\"#impl-DirtyHash-for-u64\" class=\"anchor\">§</a><h3 class=\"code-header\">impl <a class=\"trait\" href=\"polars_utils/hashing/trait.DirtyHash.html\" title=\"trait polars_utils::hashing::DirtyHash\">DirtyHash</a> for <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u64.html\">u64</a></h3></section></summary><div class=\"impl-items\"><section id=\"method.dirty_hash\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/hashing.rs.html#83\">source</a><a href=\"#method.dirty_hash\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/hashing/trait.DirtyHash.html#tymethod.dirty_hash\" class=\"fn\">dirty_hash</a>(&amp;self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u64.html\">u64</a></h4></section></div></details>","DirtyHash","polars_utils::index::IdxSize"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-FloorDivMod-for-u64\" class=\"impl\"><a class=\"src rightside\" href=\"src/polars_utils/floor_divmod.rs.html#67\">source</a><a href=\"#impl-FloorDivMod-for-u64\" class=\"anchor\">§</a><h3 class=\"code-header\">impl <a class=\"trait\" href=\"polars_utils/floor_divmod/trait.FloorDivMod.html\" title=\"trait polars_utils::floor_divmod::FloorDivMod\">FloorDivMod</a> for <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u64.html\">u64</a></h3></section></summary><div class=\"impl-items\"><section id=\"method.wrapping_floor_div_mod\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/floor_divmod.rs.html#67\">source</a><a href=\"#method.wrapping_floor_div_mod\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/floor_divmod/trait.FloorDivMod.html#tymethod.wrapping_floor_div_mod\" class=\"fn\">wrapping_floor_div_mod</a>(self, other: Self) -&gt; (Self, Self)</h4></section></div></details>","FloorDivMod","polars_utils::index::IdxSize"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-IsFloat-for-u64\" class=\"impl\"><a class=\"src rightside\" href=\"src/polars_utils/float.rs.html#44\">source</a><a href=\"#impl-IsFloat-for-u64\" class=\"anchor\">§</a><h3 class=\"code-header\">impl <a class=\"trait\" href=\"polars_utils/float/trait.IsFloat.html\" title=\"trait polars_utils::float::IsFloat\">IsFloat</a> for <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u64.html\">u64</a></h3></section></summary><div class=\"impl-items\"><section id=\"method.is_float\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/float.rs.html#4-6\">source</a><a href=\"#method.is_float\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/float/trait.IsFloat.html#method.is_float\" class=\"fn\">is_float</a>() -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section><section id=\"method.is_f32\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/float.rs.html#8-10\">source</a><a href=\"#method.is_f32\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/float/trait.IsFloat.html#method.is_f32\" class=\"fn\">is_f32</a>() -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section><section id=\"method.is_f64\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/float.rs.html#12-14\">source</a><a href=\"#method.is_f64\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/float/trait.IsFloat.html#method.is_f64\" class=\"fn\">is_f64</a>() -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section><section id=\"method.nan_value\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/float.rs.html#16-18\">source</a><a href=\"#method.nan_value\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/float/trait.IsFloat.html#method.nan_value\" class=\"fn\">nan_value</a>() -&gt; Self</h4></section><section id=\"method.is_nan\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/float.rs.html#21-26\">source</a><a href=\"#method.is_nan\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/float/trait.IsFloat.html#method.is_nan\" class=\"fn\">is_nan</a>(&amp;self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a><div class=\"where\">where\n    Self: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>,</div></h4></section><section id=\"method.is_finite\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/float.rs.html#28-33\">source</a><a href=\"#method.is_finite\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/float/trait.IsFloat.html#method.is_finite\" class=\"fn\">is_finite</a>(&amp;self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a><div class=\"where\">where\n    Self: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>,</div></h4></section></div></details>","IsFloat","polars_utils::index::IdxSize"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-IsNull-for-u64\" class=\"impl\"><a class=\"src rightside\" href=\"src/polars_utils/nulls.rs.html#56\">source</a><a href=\"#impl-IsNull-for-u64\" class=\"anchor\">§</a><h3 class=\"code-header\">impl <a class=\"trait\" href=\"polars_utils/nulls/trait.IsNull.html\" title=\"trait polars_utils::nulls::IsNull\">IsNull</a> for <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u64.html\">u64</a></h3></section></summary><div class=\"impl-items\"><section id=\"associatedconstant.HAS_NULLS\" class=\"associatedconstant trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/nulls.rs.html#56\">source</a><a href=\"#associatedconstant.HAS_NULLS\" class=\"anchor\">§</a><h4 class=\"code-header\">const <a href=\"polars_utils/nulls/trait.IsNull.html#associatedconstant.HAS_NULLS\" class=\"constant\">HAS_NULLS</a>: <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a> = false</h4></section><section id=\"associatedtype.Inner\" class=\"associatedtype trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/nulls.rs.html#56\">source</a><a href=\"#associatedtype.Inner\" class=\"anchor\">§</a><h4 class=\"code-header\">type <a href=\"polars_utils/nulls/trait.IsNull.html#associatedtype.Inner\" class=\"associatedtype\">Inner</a> = <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u64.html\">u64</a></h4></section><section id=\"method.is_null\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/nulls.rs.html#56\">source</a><a href=\"#method.is_null\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/nulls/trait.IsNull.html#tymethod.is_null\" class=\"fn\">is_null</a>(&amp;self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section><section id=\"method.unwrap_inner\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/nulls.rs.html#56\">source</a><a href=\"#method.unwrap_inner\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/nulls/trait.IsNull.html#tymethod.unwrap_inner\" class=\"fn\">unwrap_inner</a>(self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u64.html\">u64</a></h4></section></div></details>","IsNull","polars_utils::index::IdxSize"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-MinMax-for-u64\" class=\"impl\"><a class=\"src rightside\" href=\"src/polars_utils/min_max.rs.html#77\">source</a><a href=\"#impl-MinMax-for-u64\" class=\"anchor\">§</a><h3 class=\"code-header\">impl <a class=\"trait\" href=\"polars_utils/min_max/trait.MinMax.html\" title=\"trait polars_utils::min_max::MinMax\">MinMax</a> for <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u64.html\">u64</a></h3></section></summary><div class=\"impl-items\"><section id=\"method.nan_min_lt\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/min_max.rs.html#77\">source</a><a href=\"#method.nan_min_lt\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/min_max/trait.MinMax.html#tymethod.nan_min_lt\" class=\"fn\">nan_min_lt</a>(&amp;self, other: &amp;Self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section><section id=\"method.nan_max_lt\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/min_max.rs.html#77\">source</a><a href=\"#method.nan_max_lt\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/min_max/trait.MinMax.html#tymethod.nan_max_lt\" class=\"fn\">nan_max_lt</a>(&amp;self, other: &amp;Self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section><section id=\"method.min_propagate_nan\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/min_max.rs.html#19-25\">source</a><a href=\"#method.min_propagate_nan\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/min_max/trait.MinMax.html#method.min_propagate_nan\" class=\"fn\">min_propagate_nan</a>(self, other: Self) -&gt; Self</h4></section><section id=\"method.max_propagate_nan\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/min_max.rs.html#28-34\">source</a><a href=\"#method.max_propagate_nan\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/min_max/trait.MinMax.html#method.max_propagate_nan\" class=\"fn\">max_propagate_nan</a>(self, other: Self) -&gt; Self</h4></section><section id=\"method.min_ignore_nan\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/min_max.rs.html#37-43\">source</a><a href=\"#method.min_ignore_nan\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/min_max/trait.MinMax.html#method.min_ignore_nan\" class=\"fn\">min_ignore_nan</a>(self, other: Self) -&gt; Self</h4></section><section id=\"method.max_ignore_nan\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/min_max.rs.html#46-52\">source</a><a href=\"#method.max_ignore_nan\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/min_max/trait.MinMax.html#method.max_ignore_nan\" class=\"fn\">max_ignore_nan</a>(self, other: Self) -&gt; Self</h4></section></div></details>","MinMax","polars_utils::index::IdxSize"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-ToIdx-for-u64\" class=\"impl\"><a class=\"src rightside\" href=\"src/polars_utils/index.rs.html#169\">source</a><a href=\"#impl-ToIdx-for-u64\" class=\"anchor\">§</a><h3 class=\"code-header\">impl <a class=\"trait\" href=\"polars_utils/index/trait.ToIdx.html\" title=\"trait polars_utils::index::ToIdx\">ToIdx</a> for <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u64.html\">u64</a></h3></section></summary><div class=\"impl-items\"><section id=\"method.to_idx\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/index.rs.html#169\">source</a><a href=\"#method.to_idx\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/index/trait.ToIdx.html#tymethod.to_idx\" class=\"fn\">to_idx</a>(self, _len: <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u64.html\">u64</a>) -&gt; <a class=\"type\" href=\"polars_utils/index/type.IdxSize.html\" title=\"type polars_utils::index::IdxSize\">IdxSize</a></h4></section></div></details>","ToIdx","polars_utils::index::IdxSize"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-ToTotalOrd-for-u64\" class=\"impl\"><a class=\"src rightside\" href=\"src/polars_utils/total_ord.rs.html#506\">source</a><a href=\"#impl-ToTotalOrd-for-u64\" class=\"anchor\">§</a><h3 class=\"code-header\">impl <a class=\"trait\" href=\"polars_utils/total_ord/trait.ToTotalOrd.html\" title=\"trait polars_utils::total_ord::ToTotalOrd\">ToTotalOrd</a> for <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u64.html\">u64</a></h3></section></summary><div class=\"impl-items\"><section id=\"associatedtype.TotalOrdItem\" class=\"associatedtype trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/total_ord.rs.html#506\">source</a><a href=\"#associatedtype.TotalOrdItem\" class=\"anchor\">§</a><h4 class=\"code-header\">type <a href=\"polars_utils/total_ord/trait.ToTotalOrd.html#associatedtype.TotalOrdItem\" class=\"associatedtype\">TotalOrdItem</a> = <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u64.html\">u64</a></h4></section><section id=\"associatedtype.SourceItem\" class=\"associatedtype trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/total_ord.rs.html#506\">source</a><a href=\"#associatedtype.SourceItem\" class=\"anchor\">§</a><h4 class=\"code-header\">type <a href=\"polars_utils/total_ord/trait.ToTotalOrd.html#associatedtype.SourceItem\" class=\"associatedtype\">SourceItem</a> = <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u64.html\">u64</a></h4></section><section id=\"method.to_total_ord\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/total_ord.rs.html#506\">source</a><a href=\"#method.to_total_ord\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/total_ord/trait.ToTotalOrd.html#tymethod.to_total_ord\" class=\"fn\">to_total_ord</a>(&amp;self) -&gt; Self::<a class=\"associatedtype\" href=\"polars_utils/total_ord/trait.ToTotalOrd.html#associatedtype.TotalOrdItem\" title=\"type polars_utils::total_ord::ToTotalOrd::TotalOrdItem\">TotalOrdItem</a></h4></section><section id=\"method.peel_total_ord\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/total_ord.rs.html#506\">source</a><a href=\"#method.peel_total_ord\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/total_ord/trait.ToTotalOrd.html#tymethod.peel_total_ord\" class=\"fn\">peel_total_ord</a>(ord_item: Self::<a class=\"associatedtype\" href=\"polars_utils/total_ord/trait.ToTotalOrd.html#associatedtype.TotalOrdItem\" title=\"type polars_utils::total_ord::ToTotalOrd::TotalOrdItem\">TotalOrdItem</a>) -&gt; Self::<a class=\"associatedtype\" href=\"polars_utils/total_ord/trait.ToTotalOrd.html#associatedtype.SourceItem\" title=\"type polars_utils::total_ord::ToTotalOrd::SourceItem\">SourceItem</a></h4></section></div></details>","ToTotalOrd","polars_utils::index::IdxSize"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-TotalEq-for-u64\" class=\"impl\"><a class=\"src rightside\" href=\"src/polars_utils/total_ord.rs.html#254\">source</a><a href=\"#impl-TotalEq-for-u64\" class=\"anchor\">§</a><h3 class=\"code-header\">impl <a class=\"trait\" href=\"polars_utils/total_ord/trait.TotalEq.html\" title=\"trait polars_utils::total_ord::TotalEq\">TotalEq</a> for <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u64.html\">u64</a></h3></section></summary><div class=\"impl-items\"><section id=\"method.tot_eq\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/total_ord.rs.html#254\">source</a><a href=\"#method.tot_eq\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/total_ord/trait.TotalEq.html#tymethod.tot_eq\" class=\"fn\">tot_eq</a>(&amp;self, other: &amp;Self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section><section id=\"method.tot_ne\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/total_ord.rs.html#254\">source</a><a href=\"#method.tot_ne\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/total_ord/trait.TotalEq.html#method.tot_ne\" class=\"fn\">tot_ne</a>(&amp;self, other: &amp;Self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section></div></details>","TotalEq","polars_utils::index::IdxSize"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-TotalHash-for-u64\" class=\"impl\"><a class=\"src rightside\" href=\"src/polars_utils/total_ord.rs.html#254\">source</a><a href=\"#impl-TotalHash-for-u64\" class=\"anchor\">§</a><h3 class=\"code-header\">impl <a class=\"trait\" href=\"polars_utils/total_ord/trait.TotalHash.html\" title=\"trait polars_utils::total_ord::TotalHash\">TotalHash</a> for <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u64.html\">u64</a></h3></section></summary><div class=\"impl-items\"><section id=\"method.tot_hash\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/total_ord.rs.html#254\">source</a><a href=\"#method.tot_hash\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/total_ord/trait.TotalHash.html#tymethod.tot_hash\" class=\"fn\">tot_hash</a>&lt;H&gt;(&amp;self, state: <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.reference.html\">&amp;mut H</a>)<div class=\"where\">where\n    H: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/hash/trait.Hasher.html\" title=\"trait core::hash::Hasher\">Hasher</a>,</div></h4></section><section id=\"method.tot_hash_slice\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/total_ord.rs.html#79-87\">source</a><a href=\"#method.tot_hash_slice\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/total_ord/trait.TotalHash.html#method.tot_hash_slice\" class=\"fn\">tot_hash_slice</a>&lt;H&gt;(data: &amp;[Self], state: <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.reference.html\">&amp;mut H</a>)<div class=\"where\">where\n    H: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/hash/trait.Hasher.html\" title=\"trait core::hash::Hasher\">Hasher</a>,\n    Self: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sized.html\" title=\"trait core::marker::Sized\">Sized</a>,</div></h4></section></div></details>","TotalHash","polars_utils::index::IdxSize"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-TotalOrd-for-u64\" class=\"impl\"><a class=\"src rightside\" href=\"src/polars_utils/total_ord.rs.html#254\">source</a><a href=\"#impl-TotalOrd-for-u64\" class=\"anchor\">§</a><h3 class=\"code-header\">impl <a class=\"trait\" href=\"polars_utils/total_ord/trait.TotalOrd.html\" title=\"trait polars_utils::total_ord::TotalOrd\">TotalOrd</a> for <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u64.html\">u64</a></h3></section></summary><div class=\"impl-items\"><section id=\"method.tot_cmp\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/total_ord.rs.html#254\">source</a><a href=\"#method.tot_cmp\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/total_ord/trait.TotalOrd.html#tymethod.tot_cmp\" class=\"fn\">tot_cmp</a>(&amp;self, other: &amp;Self) -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/nightly/core/cmp/enum.Ordering.html\" title=\"enum core::cmp::Ordering\">Ordering</a></h4></section><section id=\"method.tot_lt\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/total_ord.rs.html#254\">source</a><a href=\"#method.tot_lt\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/total_ord/trait.TotalOrd.html#method.tot_lt\" class=\"fn\">tot_lt</a>(&amp;self, other: &amp;Self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section><section id=\"method.tot_gt\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/total_ord.rs.html#254\">source</a><a href=\"#method.tot_gt\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/total_ord/trait.TotalOrd.html#method.tot_gt\" class=\"fn\">tot_gt</a>(&amp;self, other: &amp;Self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section><section id=\"method.tot_le\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/total_ord.rs.html#254\">source</a><a href=\"#method.tot_le\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/total_ord/trait.TotalOrd.html#method.tot_le\" class=\"fn\">tot_le</a>(&amp;self, other: &amp;Self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section><section id=\"method.tot_ge\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/polars_utils/total_ord.rs.html#254\">source</a><a href=\"#method.tot_ge\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"polars_utils/total_ord/trait.TotalOrd.html#method.tot_ge\" class=\"fn\">tot_ge</a>(&amp;self, other: &amp;Self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section></div></details>","TotalOrd","polars_utils::index::IdxSize"]]]]);
    if (window.register_type_impls) {
        window.register_type_impls(type_impls);
    } else {
        window.pending_type_impls = type_impls;
    }
})()
//{"start":55,"fragment_lengths":[24262]}