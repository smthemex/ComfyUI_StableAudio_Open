# A node using Stable-audio-open-1.0 in comfyUI

## NOTICE
You can find Stable-audio-open-1.0  [Stable-audio-open-1.0](https://github.com/Stability-AI/stable-audio-tools)


1.Installation  安装   
----
 ``` python 
 https://github.com/smthemex/ComfyUI_StableAudio_Open.git
 ```
2  Dependencies  需求库  
-----
 ``` python 
pip install stable-audio-tools
 ```
这个库的需求很奇特，很大可能导致其他插件无法使用，请谨慎尝试，或者使用Conda    
The requirements for this library are very unique and may cause other plugins to be unusable. Please try it carefully or use Conda    

3 Download the checkpoints   下载模型
----
模型地址 Model address   
 [Stable-audio-open-1.0 (huggingface)](https://huggingface.co/stabilityai/stable-audio-open-1.0)  

   
4 about node  模型节点说明
----
如果使用repo_id ,不一定能连上，需要申请。如果你直接下载，有两种使用方法：  
1、是把你下载的模型文件放在comfyUI的models/diffusers 下，然后repo_id 节点就能找到，保持repo_id为空时，就可以使用。  
2、是保持上面的菜单为none（默认），在下方的地址栏填写你的模型的实际链接地址（windows系统使用"/"），例如：you_path/Stability-AI/stable-audio-tools    

If using repo_id, it may not be possible to connect and requires an application. If you download directly, there are two ways to use it:       
1. It is to place the downloaded model file in the "..models/diffusers" section of ComfyUI, and then find the repo_id node. When the repo_id is left empty, it can be used.       
2. Keep the menu above as "none" (default), and fill in the actual link address of your model in the address bar below (using "/" in Windows systems), for example: you_path/Stability-AI/stable-audio-tools       

5 example 示例
----

![](https://github.com/smthemex/ComfyUI_StableAudio_Open/blob/main/exapmle.png)


6 LICENSE 
------

``` python  
STABILITY AI NON-COMMERCIAL RESEARCH COMMUNITY LICENSE AGREEMENT 
Dated: June 5, 2024

By using or distributing any portion or element of the Models, Software, Software Products or Derivative Works, you agree to be bound by this Agreement.

"Agreement" means this Stable Non-Commercial Research Community License Agreement.

“AUP” means the Stability AI Acceptable Use Policy available at https://stability.ai/use-policy, as may be updated from time to time.

"Derivative Work(s)” means (a) any derivative work of the Software Products as recognized by U.S. copyright laws and (b) any modifications to a Model, and any other model created which is based on or derived from the Model or the Model’s output. For clarity, Derivative Works do not include the output of any Model.

“Documentation” means any specifications, manuals, documentation, and other written information provided by Stability AI related to the Software.

"Licensee" or "you" means you, or your employer or any other person or entity (if you are entering into this Agreement on such person or entity's behalf), of the age required under applicable laws, rules or regulations to provide legal consent and that has legal authority to bind your employer or such other person or entity if you are entering in this Agreement on their behalf.

“Model(s)" means, collectively, Stability AI’s proprietary models and algorithms, including machine-learning models, trained model weights and other elements of the foregoing, made available under this Agreement.

“Non-Commercial Uses” means exercising any of the rights granted herein for the purpose of research or non-commercial purposes.  For the avoidance of doubt, personal creative use is permissible as “Non-Commercial Use.” Non-Commercial Use does not, however, include the sale of Stability’s underlying Models to third parties or use of outputs from Stability’s underlying Models to train or create a competing product or service.

"Stability AI" or "we" means Stability AI Ltd. and its affiliates.

"Software" means Stability AI’s proprietary software made available under this Agreement.

“Software Products” means the Models, Software and Documentation, individually or in any combination.

1. 	License Rights and Redistribution.

a. Subject to your compliance with this Agreement, the AUP (which is hereby incorporated herein by reference), and the Documentation, Stability AI grants you a non-exclusive, worldwide, non-transferable, non-sublicensable, revocable, royalty free and limited license under Stability AI’s intellectual property or other rights owned or controlled by Stability AI embodied in the Software Products to use, reproduce, distribute, and create Derivative Works of, the Software Products, in each case for Non-Commercial Uses only, unless you subscribe to a membership via https://stability.ai/membership or otherwise obtain a commercial license from Stability AI.

b. You may not use the Software Products or Derivative Works to enable third parties to use the Software Products or Derivative Works as part of your hosted service or via your APIs, whether you are adding substantial additional functionality thereto or not. Merely distributing the Software Products or Derivative Works for download online without offering any related service (ex. by distributing the Models on HuggingFace) is not a violation of this subsection. If you wish to use the Software Products or any Derivative Works for commercial or production use or you wish to make the Software Products or any Derivative Works available to third parties via your hosted service or your APIs, contact Stability AI at https://stability.ai/contact.

c.	If you distribute or make the Software Products, or any Derivative Works thereof, available to a third party, the Software Products, Derivative Works, or any portion thereof, respectively, will remain subject to this Agreement and you must (i) provide a copy of this Agreement to such third party, and (ii) retain the following attribution notice within a "Notice" text file distributed as a part of such copies: "This Stability AI Model is licensed under the Stability AI Non-Commercial Research Community License, Copyright (c) Stability AI Ltd. All Rights Reserved.” If you create a Derivative Work of a Software Product, you may add your own attribution notices to the Notice file included with the Software Product, provided that you clearly indicate which attributions apply to the Software Product and you must state in the NOTICE file that you changed the Software Product and how it was modified.


2.	Disclaimer of Warranty. UNLESS REQUIRED BY APPLICABLE LAW, THE SOFTWARE PRODUCTS  AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE APPROPRIATENESS OF USING OR REDISTRIBUTING THE SOFTWARE PRODUCTS, DERIVATIVE WORKS OR ANY OUTPUT OR RESULTS AND ASSUME ANY RISKS ASSOCIATED WITH YOUR USE OF THE SOFTWARE PRODUCTS, DERIVATIVE WORKS AND ANY OUTPUT AND RESULTS.


3.	Limitation of Liability. IN NO EVENT WILL STABILITY AI OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, TORT, NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, ARISING OUT OF THIS AGREEMENT, FOR ANY LOST PROFITS OR ANY DIRECT, INDIRECT, SPECIAL, CONSEQUENTIAL, INCIDENTAL, EXEMPLARY OR PUNITIVE DAMAGES, EVEN IF STABILITY AI OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF ANY OF THE FOREGOING.


4.  Intellectual Property.

a. No trademark licenses are granted under this Agreement, and in connection with the Software Products or Derivative Works, neither Stability AI nor Licensee may use any name or mark owned by or associated with the other or any of its affiliates, except as required for reasonable and customary use in describing and redistributing the Software Products or Derivative Works.

b.	Subject to Stability AI’s ownership of the Software Products and Derivative Works made by or for Stability AI, with respect to any Derivative Works that are made by you, as between you and Stability AI, you are and will be the owner of such Derivative Works

c. If you institute litigation or other proceedings against Stability AI (including a cross-claim or counterclaim in a lawsuit) alleging that the Software Products, Derivative Works or associated outputs or results, or any portion of any of the foregoing, constitutes infringement of intellectual property or other rights owned or licensable by you, then any licenses granted to you under this Agreement shall terminate as of the date such litigation or claim is filed or instituted. You will indemnify and hold harmless Stability AI from and against any claim by any third party arising out of or related to your use or distribution of the Software Products or Derivative Works in violation of this Agreement.


5. 	Term and Termination. The term of this Agreement will commence upon your acceptance of this Agreement or access to the Software Products and will continue in full force and effect until terminated in accordance with the terms and conditions herein. Stability AI may terminate this Agreement if you are in breach of any term or condition of this Agreement. Upon termination of this Agreement, you shall delete and cease use of any Software Products or Derivative Works. Sections 2-4 shall survive the termination of this Agreement.

```
