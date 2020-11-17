#!/usr/bin/bash

class_name=$1

# train

mkdir -p ${class_name}/train

plink --bfile ./mdd.151.genotypes_no_missing_IDs --remove ${class_name}/test_id.txt --make-bed --out train.genotypes_no_missing_IDs
plink --bfile train.genotypes_no_missing_IDs --pheno ./covar_mds_add.txt --pheno-name Res --logistic hide-covar --covar ./covar_mds_add.txt --covar-name age,BMI,smoke,sex,HAMD --ci 0.95 --adjust --out train

plink \
    --bfile train.genotypes_no_missing_IDs \
    --clump-p1 0.005 \
    --clump-r2 0.2 \
    --clump-kb 250 \
    --clump train.assoc.logistic \
    --clump-snp-field SNP \
    --clump-field P \
    --out train

awk 'NR!=1{print $3}' train.clumped >  train.valid.snp

plink --bfile train.genotypes_no_missing_IDs --recode vcf-iid --extract train.valid.snp --out train.clump

vcftools --vcf train.clump.vcf --012 --out train.clump.snp_matrix

awk '{ $1=null;print }' train.clump.snp_matrix.012 > train.clump.snp_matrix_del0.012

paste -d" " train.clump.snp_matrix.012.indv train.clump.snp_matrix_del0.012  > train.clump.snp_matrix_del0.012_indv.txt

echo "ID" > train.clump.snpid1.txt
cat train.clump.vcf | grep -v "^#" | cut -f3 >> train.clump.snpid1.txt

cat train.clump.snpid1.txt | tr "\n" " " >train.clump.snpid2.txt
echo "" >> train.clump.snpid2.txt

cat train.clump.snpid2.txt train.clump.snp_matrix_del0.012_indv.txt > train.clump.genotype.txt

awk '{i=1;while(i <= NF){col[i]=col[i] $i " ";i=i+1}} END {i=1;while(i<=NF){print col[i];i=i+1}}' train.clump.genotype.txt | sed 's/[ \t]*$//g' > train.clump.genotype_transpose.txt

sed -i 's/  / /g' train.clump.genotype.txt

mv train* ./${class_name}/train
mv ./${class_name}/train/train.clump.genotype.txt ./${class_name}

# test

mkdir -p ${class_name}/test

plink --bfile ./mdd.151.genotypes_no_missing_IDs --keep ./${class_name}/test_id.txt --make-bed --out test.genotypes_no_missing_IDs

plink --bfile test.genotypes_no_missing_IDs --recode vcf-iid --extract ./${class_name}/train/train.valid.snp --out test.clump

vcftools --vcf test.clump.vcf --012 --out test.clump.snp_matrix

awk '{ $1=null;print }' test.clump.snp_matrix.012 > test.clump.snp_matrix_del0.012

paste -d" " test.clump.snp_matrix.012.indv test.clump.snp_matrix_del0.012  > test.clump.snp_matrix_del0.012_indv.txt

echo "ID" > test.clump.snpid1.txt
cat test.clump.vcf | grep -v "^#" | cut -f3 >> test.clump.snpid1.txt

cat test.clump.snpid1.txt | tr "\n" " " >test.clump.snpid2.txt
echo "" >> test.clump.snpid2.txt

cat test.clump.snpid2.txt test.clump.snp_matrix_del0.012_indv.txt > test.clump.genotype.txt

awk '{i=1;while(i <= NF){col[i]=col[i] $i " ";i=i+1}} END {i=1;while(i<=NF){print col[i];i=i+1}}' test.clump.genotype.txt | sed 's/[ \t]*$//g' > test.clump.genotype_transpose.txt

sed -i 's/  / /g' test.clump.genotype.txt

mv test* ./${class_name}/test
mv ./${class_name}/test/test.clump.genotype.txt ./${class_name}/


