# GKX characteristic → OSAP signal map (measured)

Method: `scripts/feature_map.py`. Mean within-month Spearman ρ over 25 months (196006..201612), same (permno, signal month). Verdict: |ρ| ≥ 0.9 same, ≥ 0.6 close variant, else none (thresholds fixed before running).

**Summary:** {'none': 36, 'close variant': 33, 'same': 25}

| gkx             | osap               |    rho | verdict       | osap_description                           | runner_up                    |
|:----------------|:-------------------|-------:|:--------------|:-------------------------------------------|:-----------------------------|
| mvel1           | Illiquidity        | -0.905 | same          | Amihud's illiquidity                       | DolVol (-0.85)               |
| beta            | BetaFP             |  0.788 | close variant | Frazzini-Pedersen Beta                     | Beta (+0.78)                 |
| betasq          | BetaFP             |  0.787 | close variant | Frazzini-Pedersen Beta                     | Beta (+0.78)                 |
| chmom           | MomRev             |  0.767 | close variant | Momentum and LT Reversal                   | MomVol (+0.63)               |
| dolvol          | DolVol             | -0.954 | same          | Past trading volume                        | Illiquidity (-0.92)          |
| idiovol         | IdioVolAHT         | -0.903 | same          | Idiosyncratic risk (AHT)                   | RealizedVol (-0.69)          |
| indmom          |                    |  0.278 | none          | Momentum (12 month)                        | IndMom (+0.26)               |
| mom1m           |                    |  0.451 | none          | 52 week high                               | ReturnSkew (-0.34)           |
| mom6m           | Mom6m              |  0.995 | same          | Momentum (6 month)                         | FirmAgeMom (+0.99)           |
| mom12m          | Mom12m             |  0.995 | same          | Momentum (12 month)                        | Mom12mOffSeason (+0.90)      |
| mom36m          | MomRev             | -0.97  | same          | Momentum and LT Reversal                   | LRreversal (-0.95)           |
| pricedelay      |                    | -0.201 | none          | Frazzini-Pedersen Beta                     | Beta (-0.19)                 |
| turn            | VolMkt             | -0.874 | close variant | Volume to market equity                    | zerotrade1M (-0.81)          |
| absacc          |                    | -0.241 | none          | Operating Cash flows to price              | IdioVolAHT (-0.24)           |
| acc             | AccrualsBM         | -0.774 | close variant | Book-to-market and accruals                | PctAcc (-0.75)               |
| age             | FirmAge            | -0.85  | close variant | Firm age based on CRSP                     | fgr5yrLag (+0.48)            |
| agr             | AssetGrowth        |  0.998 | same          | Asset growth                               | dNoa (+0.71)                 |
| bm              | BMdec              |  0.961 | same          | Book to market using December ME           | AccrualsBM (+0.92)           |
| bm_ia           |                    |  0.518 | none          | Book-to-market and accruals                | BMdec (+0.52)                |
| cashdebt        | OperProfRD         |  0.74  | close variant | Operating profitability R&D adjusted       | OScore (+0.64)               |
| cashpr          | CashProd           | -0.808 | close variant | Cash Productivity                          | AccrualsBM (-0.80)           |
| cfp             | cfp                |  0.97  | same          | Operating Cash flows to price              | PctAcc (+0.71)               |
| cfp_ia          |                    |  0.531 | none          | Operating Cash flows to price              | PctAcc (+0.38)               |
| chatoia         | ChAssetTurnover    |  0.636 | close variant | Change in Asset Turnover                   | GrSaleToGrOverhead (+0.31)   |
| chcsho          | ShareIss1Y         | -0.614 | close variant | Share issuance (1 year)                    | NetPayoutYield (-0.53)       |
| chempia         | hire               | -0.731 | close variant | Employment growth                          | AccrualsBM (-0.44)           |
| chinv           | ChInv              | -0.997 | same          | Inventory Growth                           | InvGrowth (-0.91)            |
| chpmia          |                    |  0.232 | none          | Piotroski F-score                          | EarningsConsistency (+0.18)  |
| convind         | ConvDebt           | -1     | same          | Convertible debt indicator                 | Recomm_ShortInterest (-0.31) |
| currat          |                    |  0.561 | none          | Book leverage (annual)                     | NetDebtPrice (+0.53)         |
| depr            |                    |  0.332 | none          | Net debt to price                          | GP (+0.33)                   |
| divi            |                    | -0.068 | none          | Idiosyncratic risk (AHT)                   | FirmAge (+0.07)              |
| divo            |                    | -0.118 | none          | Initial Public Offerings                   | FEPS (-0.10)                 |
| dy              | NetPayoutYield     |  0.625 | close variant | Net Payout Yield                           | fgr5yrLag (+0.59)            |
| egr             | ChEQ               | -0.999 | same          | Growth in book equity                      | DelEqu (-0.87)               |
| ep              | EP                 |  0.926 | same          | Earnings-to-Price Ratio                    | CF (+0.73)                   |
| gma             | GP                 |  0.943 | same          | gross profits / total assets               | OPLeverage (+0.60)           |
| grcapx          | grcapx             | -0.975 | same          | Change in capex (two years)                | grcapx3y (-0.83)             |
| grltnoa         | InvestPPEInv       | -0.707 | close variant | change in ppe and inv/assets               | dNoa (-0.66)                 |
| herf            |                    | -0.242 | none          | R&D over market cap                        | PredictedFE (-0.22)          |
| hire            | hire               | -0.973 | same          | Employment growth                          | AccrualsBM (-0.57)           |
| invest          | InvestPPEInv       | -0.997 | same          | change in ppe and inv/assets               | InvGrowth (-0.72)            |
| lev             | Leverage           |  0.95  | same          | Market leverage                            | AM (+0.86)                   |
| lgr             | AssetGrowth        | -0.745 | close variant | Asset growth                               | DelFINL (-0.71)              |
| mve_ia          |                    | -0.487 | none          | Brand capital investment                   | RIO_MB (-0.46)               |
| operprof        | OperProf           |  0.96  | same          | operating profits / book equity            | OperProfRD (+0.78)           |
| orgcap          | OrgCap             |  0.798 | close variant | Organizational capital                     | GP (+0.74)                   |
| pchcapx_ia      |                    | -0.459 | none          | Investment to revenue                      | ChInvIA (-0.46)              |
| pchcurrat       |                    |  0.355 | none          | Change in current operating liabilities    | PS (+0.32)                   |
| pchdepr         |                    |  0.392 | none          | Investment to revenue                      | grcapx3y (+0.34)             |
| pchgm_pchsale   |                    |  0.237 | none          | Piotroski F-score                          | OperProfRD (+0.22)           |
| pchquick        |                    |  0.301 | none          | Change in current operating liabilities    | PS (+0.29)                   |
| pchsale_pchinvt | GrSaleToGrInv      |  0.868 | close variant | Sales growth over inventory growth         | InvGrowth (+0.64)            |
| pchsale_pchrect |                    |  0.321 | none          | Change in current operating assets         | AccrualsBM (+0.32)           |
| pchsale_pchxsga | GrSaleToGrOverhead |  0.875 | close variant | Sales growth over overhead growth          | ChAssetTurnover (+0.33)      |
| pchsaleinv      | GrSaleToGrInv      |  0.862 | close variant | Sales growth over inventory growth         | InvGrowth (+0.65)            |
| pctacc          | PctAcc             | -0.999 | same          | Percent Operating Accruals                 | AccrualsBM (-0.72)           |
| ps              | PS                 |  0.787 | close variant | Piotroski F-score                          | OperProfRD (+0.37)           |
| quick           |                    |  0.539 | none          | Net debt to price                          | Cash (+0.53)                 |
| rd              |                    |  0.542 | none          | R&D capital-to-assets                      | SurpriseRD (+0.50)           |
| rd_mve          | RD                 |  0.956 | same          | R&D over market cap                        | RDcap (+0.86)                |
| rd_sale         | RDcap              |  0.89  | close variant | R&D capital-to-assets                      | RD (+0.74)                   |
| realestate      | realestate         |  0.856 | close variant | Real estate holdings                       | RD (-0.27)                   |
| roic            | OperProfRD         |  0.858 | close variant | Operating profitability R&D adjusted       | OperProf (+0.74)             |
| salecash        | Cash               | -0.694 | close variant | Cash to assets                             | SP (+0.54)                   |
| saleinv         |                    | -0.274 | none          | R&D over market cap                        | sinAlgo (-0.26)              |
| salerec         |                    |  0.48  | none          | Operating leverage                         | AdExp (+0.32)                |
| secured         |                    |  0.409 | none          | Brand capital investment                   | Illiquidity (+0.36)          |
| securedind      |                    | -0.287 | none          | Debt Issuance                              | NOA (-0.25)                  |
| sgr             | AccrualsBM         | -0.636 | close variant | Book-to-market and accruals                | AssetGrowth (-0.56)          |
| sin             | sinAlgo            |  0.701 | close variant | Sin Stock (selection criteria)             | ProbInformedTrading (-0.13)  |
| sp              | SP                 |  0.931 | same          | Sales-to-price                             | AccrualsBM (+0.68)           |
| tang            | tang               |  0.666 | close variant | Tangibility                                | Cash (+0.53)                 |
| tb              | Tax                |  0.796 | close variant | Taxable income to income                   | OperProfRD (+0.33)           |
| aeavol          |                    | -0.21  | none          | Analyst Recommendations and Short-Interest | GP (+0.12)                   |
| cash            | Cash               |  0.868 | close variant | Cash to assets                             | NetDebtPrice (+0.61)         |
| chtx            |                    |  0.342 | none          | Change in Taxes                            | AccrualsBM (-0.29)           |
| cinvest         |                    |  0.089 | none          | change in ppe and inv/assets               | DelFINL (+0.07)              |
| ear             |                    |  0.195 | none          | Momentum without the seasonal part         | Mom12m (+0.20)               |
| nincr           |                    |  0.186 | none          | Earnings surprise streak                   | AccrualsBM (-0.17)           |
| roaq            | OperProfRD         |  0.652 | close variant | Operating profitability R&D adjusted       | roaq (+0.65)                 |
| roavol          |                    | -0.541 | none          | Idiosyncratic risk (AHT)                   | FEPS (-0.46)                 |
| roeq            | RoE                |  0.66  | close variant | net income / book equity                   | OperProf (+0.60)             |
| rsup            |                    | -0.312 | none          | Asset growth                               | DelCOA (-0.30)               |
| stdacc          |                    | -0.341 | none          | Operating leverage                         | IdioVolAHT (-0.34)           |
| stdcf           |                    | -0.367 | none          | Operating leverage                         | FEPS (-0.34)                 |
| ms              |                    |  0.571 | none          | Operating profitability R&D adjusted       | CBOperProf (+0.46)           |
| baspread        | IdioVolAHT         | -0.674 | close variant | Idiosyncratic risk (AHT)                   | RealizedVol (-0.67)          |
| ill             | Illiquidity        |  0.935 | same          | Amihud's illiquidity                       | DolVol (+0.88)               |
| maxret          | MaxRet             | -1     | same          | Maximum return over month                  | RealizedVol (-0.89)          |
| retvol          | RealizedVol        | -1     | same          | Realized (Total) Volatility                | IdioVol3F (-0.97)            |
| std_dolvol      |                    |  0.582 | none          | Amihud's illiquidity                       | DolVol (+0.52)               |
| std_turn        | VolMkt             | -0.69  | close variant | Volume to market equity                    | std_turn (-0.58)             |
| zerotrade       | zerotrade1M        |  0.843 | close variant | Days with zero trades                      | zerotrade6M (+0.82)          |
