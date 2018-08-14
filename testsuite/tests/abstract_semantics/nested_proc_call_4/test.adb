procedure Test is
   A : Integer;

   procedure P is
      B : Integer;

      procedure Q is
      begin
         A := 31;
         B := 14;
      end Q;
   begin
      Q;
   end P;
begin
   P;
end Ex1;
