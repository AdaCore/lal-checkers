procedure Test is
   procedure F(R : out Integer; X : Integer := 0) is
   begin
      null;
   end F;

   R : Integer;
begin
   F(R);
end Ex1;
