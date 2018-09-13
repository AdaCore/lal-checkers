procedure Test is
   generic
      with procedure F;
   procedure Iterate;

   Found : Boolean := False;

   procedure My_F is
   begin
      Found := True;
   end My_F;

   procedure My_Iterate is new Iterate(F => My_F);
begin
   My_Iterate;
end Test;
